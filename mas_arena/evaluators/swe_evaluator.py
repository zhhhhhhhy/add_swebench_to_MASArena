from typing import Dict, Any, Tuple
from mas_arena.evaluators.base_evaluator import BaseEvaluator
from mas_arena.evaluators.registry import register_benchmark
from langsmith.evaluation import RunEvaluator
from langsmith.schemas import Run
import json
import os
import docker 
from pathlib import Path, PurePosixPath
import platform
if platform.system() == "Linux":
    import resource
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import hashlib

import traceback
from dataclasses import dataclass
from typing import Optional, Union, cast
from .constants import (
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
    DOCKER_PATCH,
    DOCKER_USER,
    DOCKER_WORKDIR,
    INSTANCE_IMAGE_BUILD_DIR,
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    LOG_REPORT,
    LOG_INSTANCE,
    LOG_TEST_OUTPUT,
    RUN_EVALUATION_LOG_DIR,
    UTF8,
)
from .docker_utils import (
    clean_images,
    cleanup_container,
    copy_to_container,
    exec_run_with_timeout,
    list_images,
    remove_image,
    should_remove,
)
from .docker_build import (
    BuildImageError,
    build_container,
    build_env_images,
    close_logger,
    setup_logger,
)
from .grading import get_eval_report
from .reporting import make_run_report

from .test_spec.test_spec import make_test_spec, TestSpec
from .utils1 import (
    EvaluationError,
    load_swebench_dataset,
    get_predictions_from_file,
    run_threadpool,
    str2bool,
    optional_str,
)
GIT_APPLY_CMDS = [
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i",
]


@register_benchmark(
    name="swe",
    normalization_keys={
        "id":"instance_id",
        "repo":"repo",
        "problem": "problem_statement",
        "solution": "patch",
    }
)
# 这个evaluator只需要说能几桶一个evaluate方法，这个方法能够对 大模型生成的答案进行评估即可
class SWEEvaluator(BaseEvaluator):
    """
    Evaluator for SWE-bench problems
    
    This evaluator supports different agent system output formats and abstracts away the 
    implementation details of applying/testing patches to repositories.
    """
    def __init__(self, name: str = "swe", config: Dict[str, Any] = None):
        """
        Initialize the SWE Evaluator.

        Args:
            name: Name of the evaluator (default: "bbh")
            config: Configuration parameters
        """
        super().__init__(name, config)
        self.run_evaluator = RunEvaluator()
        
    @classmethod
    def from_config(cls, name: str, config: Dict[str, Any] = None):
        return cls(name, config)
    


    
    def get_dataset_from_preds(self,instance_id):
        # Construct the path to the JSONL file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, '..', '..', 'data', 'swe_test.jsonl')
        
        try:
            with open(data_path, 'r', encoding='utf-8') as file:
                for line in file:
                    # Parse each line as a JSON object
                    record = json.loads(line.strip())
                    # Check if the instance_id matches
                    if record.get('instance_id') == instance_id:
                        return record
            return None  # Return None if no matching record is found
        except FileNotFoundError:
            print(f"Error: File {data_path} not found.")
            return None
        except json.JSONDecodeError:
            print("Error: Invalid JSON format in the file.")
            return None
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return None
    # 对单个SWE问题进行完整评测流程（提取代码→运行测试→打分→生成 Run）
    def evaluate(self, problem: Dict[str, Any], run_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a problem given the agent's response.
        
        Args:
            problem: The problem dictionary with "problem" and "solution" keys
            run_result: The result from running the agent system, including messages
            
        Returns:
            Evaluation results dictionary
        """
        # 1、获取id
        instance_id = problem["id"]
        print("instance_id:",instance_id)
        # 2、加载数据集 并 获取数据集中的其它数据
        data = self.get_dataset_from_preds(instance_id)
        print(data["instance_id"])
        # 3. 准备评估参数
        run_id = "test_run"  # 固定 run_id，可根据需要从 run_result 动态获取
        timeout = 1800
        namespace = "swebench"
        instance_image_tag = "latest"
        rewrite_reports = False
        force_rebuild = False
        print("111111111111111111111111111111")
        # 4. 构造 TestSpec 和 prediction
        client = docker.from_env()
        test_spec = make_test_spec(data, namespace=namespace, instance_image_tag=instance_image_tag)
        prediction = {
            "instance_id": instance_id,
            "model_name_or_path": "default_model",
            "model_patch": run_result.get("patch", problem.get("solution", ""))
        }
        print("222222222222222222222222222222222")
        # 5、进行评测的整体逻辑
        instance_id, report = self.run_instance(
            test_spec=test_spec,
            pred=prediction,
            rm_image=True,
            force_rebuild=force_rebuild,
            client=client,
            run_id=run_id,
            timeout=timeout,
            rewrite_reports=rewrite_reports,
        )
        print("3333333333333333333333333333333")
        # 6. 构造返回结果
        resolved = report.get(instance_id, {}).get("resolved", False)
        print("resolved:",resolved)
        return {
            "final_answer": prediction["model_patch"],
            "score": 1.0 if resolved else 0.0,
            "extracted_answer": report,
            "instance_id": instance_id
        }    
              
            
            
            
        # 5、获取评测结果    
        # 6、输出结果    
        # 该方法的返回格式为 可以增加字段
        # return {
        #     "final_answer": final_answer,
        #     "score": score,
        #     "extracted_answer": extracted_answer
        # }
        print(problem)
        
    
    def run_instance(self,test_spec,pred,rm_image,force_rebuild,client,run_id,timeout=None,
        rewrite_reports=False,
        ):
        print("4444444444444444444444444444")
        instance_id = test_spec.instance_id
        model_name_or_path = pred.get("model_name_or_path", "None").replace("/", "__")
        log_dir = RUN_EVALUATION_LOG_DIR / run_id / model_name_or_path / instance_id
        report_path = log_dir / LOG_REPORT
        print("55555555555555555555555555555")
        if rewrite_reports:
            test_output_path = log_dir / LOG_TEST_OUTPUT
            if not test_output_path.exists():
                raise ValueError(f"Test output file {test_output_path} does not exist")
            report = get_eval_report(
                test_spec=test_spec,
                prediction=pred,
                test_log_path=test_output_path,
                include_tests_status=True,
            )
            with open(report_path, "w") as f:
                f.write(json.dumps(report, indent=4))
            return instance_id, report
        print("66666666666666666666666666666666")
        if report_path.exists():
            return instance_id, json.loads(report_path.read_text())
        print("66666666666666666666666666666666")
        log_dir.mkdir(parents=True, exist_ok=True)
        container = None
        print("66666666666666666666666666666666")
        # Set up logger
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / LOG_INSTANCE
        logger = setup_logger(instance_id, log_file)
        try:
            print("Building container...")
            container = build_container(test_spec, 
                                        client,
                                        run_id, 
                                        logger,
                                        rm_image, 
                                        force_rebuild)
            print(f"Container returned: {container}, Type: {type(container)}")
            print("Starting container...")
            container.start()
            print("77777777777777777777777777777777")
        except Exception as e:
            print(f"Error in building or starting container: {e}")
            raise

        try:
            # Step 2: Prepare and copy patch file
            print("Preparing patch file...")
            patch_file = Path(log_dir / "patch.diff")
            patch_file.write_text(pred.get("model_patch", ""))
            print("Copying patch to container...")
            copy_to_container(container, patch_file, PurePosixPath(DOCKER_PATCH))
            print("8888888888888888888888888888888")
        except Exception as e:
            print(f"Error in preparing or copying patch file: {e}")
            raise
        try:
            # Step 3: Apply patch
            print("Applying patch...")
            applied_patch = False
            for git_apply_cmd in GIT_APPLY_CMDS:
                val = container.exec_run(
                    f"{git_apply_cmd} {DOCKER_PATCH}",
                    workdir=DOCKER_WORKDIR,
                    user=DOCKER_USER,
                )
                if val.exit_code == 0:
                    applied_patch = True
                    break
            if not applied_patch:
                raise ValueError(f"{APPLY_PATCH_FAIL}:\n{val.output.decode(UTF8)}")
            print("9999999999999999999999999999")
        except Exception as e:
            print(f"Error in applying patch: {e}")
            raise

        try:
            # Step 4: Get git diff before
            print("Getting git diff before...")
            git_diff_output_before = (
                container.exec_run("git -c core.fileMode=false diff", workdir=DOCKER_WORKDIR)
                .output.decode(UTF8)
                .strip()
            )
            print("ttttttttttttttttttttttttttttt")
        except Exception as e:
            print(f"Error in getting git diff before: {e}")
            raise

        try:
            # Step 5: Prepare and copy eval script
            print("Preparing eval script...")
            eval_file = Path(log_dir / "eval.sh")
            eval_file.write_text(test_spec.eval_script)
            print("Copying eval script to container...")
            copy_to_container(container, eval_file, PurePosixPath("/eval.sh"))
            print("rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr")
        except Exception as e:
            print(f"Error in preparing or copying eval script: {e}")
            raise
        try:
            # Step 6: Run eval script
            print("Running eval script...")
            test_output, timed_out, _ = exec_run_with_timeout(
                container, "/bin/bash /eval.sh", timeout
            )
            test_output_path = log_dir / LOG_TEST_OUTPUT
            with open(test_output_path, "w") as f:
                f.write(test_output)
                if timed_out:
                    f.write(f"\n\nTimeout error: {timeout} seconds exceeded.")
                    raise ValueError(f"Test timed out after {timeout} seconds.")
            print("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
        except Exception as e:
            print(f"Error in running eval script: {e}")
            raise

        try:
            # Step 7: Get git diff after
            print("Getting git diff after...")
            git_diff_output_after = (
                container.exec_run("git -c core.fileMode=false diff", workdir=DOCKER_WORKDIR)
                .output.decode(UTF8)
                .strip()
            )
            print("wwwwwwwwwwwwwwwwwwwwwwwwwwwwww")
        except Exception as e:
            print(f"Error in getting git diff after: {e}")
            raise

        try:
            # Step 8: Generate report
            print("Generating report...")
            report = get_eval_report(
                test_spec=test_spec,
                prediction=pred,
                test_log_path=test_output_path,
                include_tests_status=True,
            )
            with open(report_path, "w") as f:
                f.write(json.dumps(report, indent=4))
                print("xxxxxxxxxxxxxxxxxxxx")
            print("xxxxxxxxxxxxxxxxxxxxinstance_id",instance_id)
            print("report qqqqqqqqqqqqqqqqqqq",report)
            return instance_id, report
        except Exception as e:
            print(f"Error in generating report: {e}")
            raise
        finally:
            print("qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq")
            cleanup_container(client, container,logger)
            print("qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq")
            if rm_image:
                remove_image(client, test_spec.instance_image_key)
            print("qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq")
        # try:
        #     container = build_container(test_spec, client, run_id, rm_image, force_rebuild)
        #     container.start()
        #     print("77777777777777777777777777777777")
        #     patch_file = Path(log_dir / "patch.diff")
        #     patch_file.write_text(pred.get("model_patch", ""))
        #     copy_to_container(container, patch_file, PurePosixPath(DOCKER_PATCH))
        #     print("8888888888888888888888888888888")
        #     applied_patch = False
        #     for git_apply_cmd in GIT_APPLY_CMDS:
        #         val = container.exec_run(
        #             f"{git_apply_cmd} {DOCKER_PATCH}",
        #             workdir=DOCKER_WORKDIR,
        #             user=DOCKER_USER,
        #         )
        #         if val.exit_code == 0:
        #             applied_patch = True
        #             break
        #     if not applied_patch:
        #         raise ValueError(f"{APPLY_PATCH_FAIL}:\n{val.output.decode(UTF8)}")
        #     print("9999999999999999999999999999")
        #     git_diff_output_before = (
        #         container.exec_run("git -c core.fileMode=false diff", workdir=DOCKER_WORKDIR)
        #         .output.decode(UTF8)
        #         .strip()
        #     )
        #     print("ttttttttttttttttttttttttttttt")
        #     eval_file = Path(log_dir / "eval.sh")
        #     eval_file.write_text(test_spec.eval_script)
        #     copy_to_container(container, eval_file, PurePosixPath("/eval.sh"))
        #     print("rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr")
        #     test_output, timed_out, _ = exec_run_with_timeout(
        #         container, "/bin/bash /eval.sh", timeout
        #     )
        #     test_output_path = log_dir / LOG_TEST_OUTPUT
        #     with open(test_output_path, "w") as f:
        #         f.write(test_output)
        #         if timed_out:
        #             f.write(f"\n\nTimeout error: {timeout} seconds exceeded.")
        #             raise ValueError(f"Test timed out after {timeout} seconds.")
        #     print("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
        #     git_diff_output_after = (
        #         container.exec_run("git -c core.fileMode=false diff", workdir=DOCKER_WORKDIR)
        #         .output.decode(UTF8)
        #         .strip()
        #     )
        #     print("wwwwwwwwwwwwwwwwwwwwwwwwwwwwww")
        #     report = get_eval_report(
        #         test_spec=test_spec,
        #         prediction=pred,
        #         test_log_path=test_output_path,
        #         include_tests_status=True,
        #     )
        #     with open(report_path, "w") as f:
        #         f.write(json.dumps(report, indent=4))
        #     return instance_id, report
        # finally:
        #     print("qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq")
        #     cleanup_container(client, container)
        #     if rm_image:
        #         remove_image(client, test_spec.instance_image_key)
# | 方法名                                   | 作用（一句话）                                       | 输入                                                                                  | 输出                                 |
# | ------------------------------------- | --------------------------------------------- | ----------------------------------------------------------------------------------- | ---------------------------------- |
# | `__init__`                            | 初始化 HumanEval 评测器，注册 LangSmith 包装器            | `name`, `config`                                                                    | 无                                  |
# | `extract_code`                        | 从模型返回文本中提取可执行 Python 代码                       | `text:str` – 模型返回的文本                                                                | `str` – 提取出的 Python 代码             |
# | `check_solution`                      | 在隔离环境中编译并运行候选函数+官方测试，判断是否全部通过                 | • `code:str` – 候选函数<br>• `test:str` – 官方测试代码<br>• `entry_point:str` – 函数名           | `(bool,str)` – 是否通过 + 报错/成功信息      |
# | `calculate_score`                     | 调用 `check_solution` 并返回 0/1 分数及附带信息           | • `test_code:str`<br>• `prediction:str`<br>• `entry_point:str`                      | `(float,str,str)` – 分数、代码、信息       |
# | `create_run`                          | 把一次评测结果打包成 LangSmith 的 `Run` 对象               | • `problem`<br>• `final_answer`<br>• `extracted_answer`<br>• `score`<br>• `message` | `Run` – LangSmith 所需结构             |
# | `evaluate`                            | 对单个 HumanEval 问题进行完整评测流程（提取代码→运行测试→打分→生成 Run） | • `problem:dict`<br>• `run_result:dict`                                             | `dict` – 含分数、代码、消息、LangSmith Run 等 |
# | `async_evaluate`                      | 把 `evaluate` 包装成异步版本，内部用线程池                   | 同 `evaluate`                                                                        | 同 `evaluate`（awaitable）            |
# | `extract_test_cases_with_entry_point` | 按函数名返回对应的测试代码（支持硬编码映射，回退到数据集）                 | `entry_point:str`                                                                   | `str` – 测试代码；找不到时返回 `None`         |

# 已知的是其传递过来的参数只有ID problem 和 solution并且只传递到evaluate这个方法中。
# 1、搜索数据库获取其他有用信息

# 2、实现evaluator

# {
#     'id': 'scikit-learn__scikit-learn-25299',
#     'problem': 'BUG log_loss renormalizes the predictions\n### Describe the bug\n\n`log_loss(y_true, y_pred)` renormalizes `y_pred` internally such that it sums to 1. This way, a really bad model, the predictions of which do not sum to 1, gets a better loss then it actually has.\n\n### Steps/Code to Reproduce\n\n```python\r\nfrom scipy.special import xlogy\r\nfrom sklearn.metrics import log_loss\r\n\r\ny_true = [[0, 1]]\r\ny_pred = [[0.2, 0.3]]\r\n\r\nlog_loss(y_true, y_pred)\r\n```\n\n### Expected Results\n\n```python\r\n-xlogy(y_true, y_pred).sum(axis=1)\r\n```\r\nResult: `1.2039728`\n\n### Actual Results\n\nResult: `0.5108256237659907`\n\n### Versions\n\n```shell\nSystem:\r\n    python: 3.9.14\r\n   machine: macOS\r\n\r\nPython dependencies:\r\n      sklearn: 1.1.2\n```\n\n',
#     'solution': 'diff --git a/sklearn/metrics/_classification.py b/sklearn/metrics/_classification.py\n--- a/sklearn/metrics/_classification.py\n+++ b/sklearn/metrics/_classification.py\n@@ -2622,6 +2622,9 @@ def log_loss(\n            The default value changed from `1e-15` to `"auto"` that is\n            equivalent to `np.finfo(y_pred.dtype).eps`.\n \n+        .. deprecated:: 1.3\n+           `eps` is deprecated in 1.3 and will be removed in 1.5.\n+\n     normalize : bool, default=True\n         If true, return the mean loss per sample.\n         Otherwise, return the sum of the per-sample losses.\n@@ -2660,7 +2663,16 @@ def log_loss(\n     y_pred = check_array(\n         y_pred, ensure_2d=False, dtype=[np.float64, np.float32, np.float16]\n     )\n-    eps = np.finfo(y_pred.dtype).eps if eps == "auto" else eps\n+    if eps == "auto":\n+        eps = np.finfo(y_pred.dtype).eps\n+    else:\n+        # TODO: Remove user defined eps in 1.5\n+        warnings.warn(\n+            "Setting the eps parameter is deprecated and will "\n+            "be removed in 1.5. Instead eps will always have"\n+            "a default value of `np.finfo(y_pred.dtype).eps`.",\n+            FutureWarning,\n+        )\n \n     check_consistent_length(y_pred, y_true, sample_weight)\n     lb = LabelBinarizer()\n@@ -2723,6 +2735,12 @@ def log_loss(\n \n     # Renormalize\n     y_pred_sum = y_pred.sum(axis=1)\n+    if not np.isclose(y_pred_sum, 1, rtol=1e-15, atol=5 * eps).all():\n+        warnings.warn(\n+            "The y_pred values do not sum to one. Starting from 1.5 this"\n+            "will result in an error.",\n+            UserWarning,\n+        )\n     y_pred = y_pred / y_pred_sum[:, np.newaxis]\n     loss = -xlogy(transformed_labels, y_pred).sum(axis=1)\n \n'
# }

# {
#     'id': 'sympy__sympy-19007',
#     'problem': "Wrong matrix element fetched from BlockMatrix\nGiven this code:\r\n```\r\nfrom sympy import *\r\nn, i = symbols('n, i', integer=True)\r\nA = MatrixSymbol('A', 1, 1)\r\nB = MatrixSymbol('B', n, 1)\r\nC = BlockMatrix([[A], [B]])\r\nprint('C is')\r\npprint(C)\r\nprint('C[i, 0] is')\r\npprint(C[i, 0])\r\n```\r\nI get this output:\r\n```\r\nC is\r\n⎡A⎤\r\n⎢ ⎥\r\n⎣B⎦\r\nC[i, 0] is\r\n(A)[i, 0]\r\n```\r\n`(A)[i, 0]` is the wrong here. `C[i, 0]` should not be simplified as that element may come from either `A` or `B`.\n",
#     'solution': "diff --git a/sympy/matrices/expressions/blockmatrix.py b/sympy/matrices/expressions/blockmatrix.py\n--- a/sympy/matrices/expressions/blockmatrix.py\n+++ b/sympy/matrices/expressions/blockmatrix.py\n@@ -7,7 +7,7 @@\n from sympy.utilities import sift\n from sympy.utilities.misc import filldedent\n \n-from sympy.matrices.expressions.matexpr import MatrixExpr, ZeroMatrix, Identity\n+from sympy.matrices.expressions.matexpr import MatrixExpr, ZeroMatrix, Identity, MatrixElement\n from sympy.matrices.expressions.matmul import MatMul\n from sympy.matrices.expressions.matadd import MatAdd\n from sympy.matrices.expressions.matpow import MatPow\n@@ -234,16 +234,24 @@ def transpose(self):\n \n     def _entry(self, i, j, **kwargs):\n
# # Find row entry\n+        orig_i, orig_j = i, j\n         for row_block, numrows in enumerate(self.rowblocksizes):\n-            if (i < numrows) != False:\n+            cmp = i < numrows\n+
#  if cmp == True:\n                 break\n-            else:\n+            elif cmp == False:\n      
#            i -= numrows\n+            elif row_block < self.blockshape[0] - 1:\n+                # Can't tell which block and it's not the last one, return unevaluated\n+                return MatrixElement(self, orig_i, orig_j)\n         for col_block, numcols in enumerate(self.colblocksizes):\n-     
#        if (j < numcols) != False:\n+            cmp = j < numcols\n+            if cmp == True:\n    
#              break\n-            else:\n+            elif cmp == False:\n                 j -= numcols\n+            elif col_block < self.blockshape[1] - 1:\n+                return MatrixElement(self, orig_i, orig_j)\n         return self.blocks[row_block, col_block][i, j]\n \n     @property\n"}