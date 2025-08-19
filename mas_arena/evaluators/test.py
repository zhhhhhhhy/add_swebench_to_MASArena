# import os
# import json
# def retrieve_by_instance_id(instance_id):
#         # Construct the path to the JSONL file
#         current_dir = os.path.dirname(os.path.abspath(__file__))
#         data_path = os.path.join(current_dir, '..', '..', 'data', 'swe_test.jsonl')
        
#         try:
#             with open(data_path, 'r', encoding='utf-8') as file:
#                 for line in file:
#                     # Parse each line as a JSON object
#                     record = json.loads(line.strip())
#                     # Check if the instance_id matches
#                     if record.get('instance_id') == instance_id:
#                         return record
#             return None  # Return None if no matching record is found
#         except FileNotFoundError:
#             print(f"Error: File {data_path} not found.")
#             return None
#         except json.JSONDecodeError:
#             print("Error: Invalid JSON format in the file.")
#             return None
#         except Exception as e:
#             print(f"Error occurred: {str(e)}")
#             return None
    
# # Example usage
# if __name__ == "__main__":
#     target_id = "scikit-learn__scikit-learn-25299"  # Replace with the instance_id you want to search for
#     result = retrieve_by_instance_id(target_id)
#     if result:
#         # print("Found record:", result)
#         print("Found record:", result["repo"])
#     else:
#         print(f"No record found for instance_id: {target_id}")
        
# from __future__ import annotations

# import docker
# import json
# import platform
# import traceback

# if platform.system() == "Linux":
#     import resource

# from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
# from pathlib import Path, PurePosixPath

# from swebench.harness.constants import (
#     APPLY_PATCH_FAIL,
#     APPLY_PATCH_PASS,
#     DOCKER_PATCH,
#     DOCKER_USER,
#     DOCKER_WORKDIR,
#     INSTANCE_IMAGE_BUILD_DIR,
#     KEY_INSTANCE_ID,
#     KEY_MODEL,
#     KEY_PREDICTION,
#     LOG_REPORT,
#     LOG_INSTANCE,
#     LOG_TEST_OUTPUT,
#     RUN_EVALUATION_LOG_DIR,
#     UTF8,
# )
# from swebench.harness.docker_utils import (
#     clean_images,
#     cleanup_container,
#     copy_to_container,
#     exec_run_with_timeout,
#     list_images,
#     remove_image,
#     should_remove,
# )
# from swebench.harness.docker_build import (
#     BuildImageError,
#     build_container,
#     build_env_images,
#     close_logger,
#     setup_logger,
# )
# from swebench.harness.grading import get_eval_report
# from swebench.harness.reporting import make_run_report
# from swebench.harness.modal_eval import (
#     run_instances_modal,
#     validate_modal_credentials,
# )
# from swebench.harness.test_spec.test_spec import make_test_spec, TestSpec
# from swebench.harness.utils import (
#     EvaluationError,
#     load_swebench_dataset,
#     get_predictions_from_file,
#     run_threadpool,
#     str2bool,
#     optional_str,
# )

# GIT_APPLY_CMDS = [
#     "git apply --verbose",
#     "git apply --verbose --reject",
#     "patch --batch --fuzz=5 -p1 -i",
# ]

# # 这个函数 run_instance 的作用是：在 Docker 容器中，针对一个具体的 SWE-bench 实例运行模型生成的补丁，并生成评估报告。
# def run_instance(test_spec: TestSpec,pred: dict,rm_image: bool,force_rebuild: bool,client: docker.DockerClient,run_id: str,timeout: int | None = None,
# ):
#     """
#     Run a single instance with the given prediction.

#     Args:
#         test_spec (TestSpec): TestSpec instance
#         pred (dict): Prediction w/ model_name_or_path, model_patch, instance_id
#         rm_image (bool): Whether to remove the image after running
#         force_rebuild (bool): Whether to force rebuild the image
#         client (docker.DockerClient): Docker client
#         run_id (str): Run ID
#         timeout (int): Timeout for running tests
#     """
#     # Set up logging directory
#     instance_id = test_spec.instance_id
#     model_name_or_path = pred.get(KEY_MODEL, "None").replace("/", "__")
    


#     if report_path.exists():
#         return instance_id, json.loads(report_path.read_text())

#     if not test_spec.is_remote_image:
#         # Link the image build dir in the log dir
#         build_dir = INSTANCE_IMAGE_BUILD_DIR / test_spec.instance_image_key.replace(
#             ":", "__"
#         )
#         image_build_link = log_dir / "image_build_dir"
#         if not image_build_link.exists():
#             try:
#                 # link the image build dir in the log dir
#                 image_build_link.symlink_to(
#                     build_dir.absolute(), target_is_directory=True
#                 )
#             except:
#                 # some error, idk why
#                 pass


#     # Run the instance
#     container = None
#     try:
#         # Build + start instance container (instance image should already be built)
#         container = build_container(
#             test_spec, client, run_id, logger, rm_image, force_rebuild
#         )
#         container.start()
       

#         # Copy model prediction as patch file to container
#         patch_file = Path(log_dir / "patch.diff")
#         patch_file.write_text(pred[KEY_PREDICTION] or "")
        
#         copy_to_container(container, patch_file, PurePosixPath(DOCKER_PATCH))

#         # Attempt to apply patch to container (TODO: FIX THIS)
#         applied_patch = False
#         for git_apply_cmd in GIT_APPLY_CMDS:
#             val = container.exec_run(
#                 f"{git_apply_cmd} {DOCKER_PATCH}",
#                 workdir=DOCKER_WORKDIR,
#                 user=DOCKER_USER,
#             )
#             if val.exit_code == 0:
#                 applied_patch = True
#                 break
            

#         # Get git diff before running eval script
#         git_diff_output_before = (
#             container.exec_run(
#                 "git -c core.fileMode=false diff", workdir=DOCKER_WORKDIR
#             )
#             .output.decode(UTF8)
#             .strip()
#         )
        

#         eval_file = Path(log_dir / "eval.sh")
#         eval_file.write_text(test_spec.eval_script)
        
#         copy_to_container(container, eval_file, PurePosixPath("/eval.sh"))

#         # Run eval script, write output to logs
#         test_output, timed_out, total_runtime = exec_run_with_timeout(
#             container, "/bin/bash /eval.sh", timeout
#         )
#         test_output_path = log_dir / LOG_TEST_OUTPUT
#         with open(test_output_path, "w") as f:
#             f.write(test_output)
#             if timed_out:
#                 f.write(f"\n\nTimeout error: {timeout} seconds exceeded.")


#         # Get git diff after running eval script (ignore permission changes)
#         git_diff_output_after = (
#             container.exec_run(
#                 "git -c core.fileMode=false diff", workdir=DOCKER_WORKDIR
#             )
#             .output.decode(UTF8)
#             .strip()
#         )

#         # Get report from test output
#         report = get_eval_report(
#             test_spec=test_spec,
#             prediction=pred,
#             test_log_path=test_output_path,
#             include_tests_status=True,
#         )

#         # Write report to report.json
#         with open(report_path, "w") as f:
#             f.write(json.dumps(report, indent=4))
#         return instance_id, report
#     finally:
#         # Remove instance container + image, close logger
#         cleanup_container(client, container, logger)
#         if rm_image:
#             remove_image(client, test_spec.instance_image_key, logger)
#     return


# def main(
#     dataset_name: str,
#     split: str,
#     instance_ids: list,
#     predictions_path: str,
#     max_workers: int,
#     force_rebuild: bool,
#     cache_level: str,
#     clean: bool,
#     open_file_limit: int,
#     run_id: str,
#     timeout: int,
#     namespace: str | None,
#     rewrite_reports: bool,
#     modal: bool,
#     instance_image_tag: str = "latest",
#     report_dir: str = ".",
# ):
#     """
#     Run evaluation harness for the given dataset and predictions.
#     """
#     if dataset_name == "SWE-bench/SWE-bench_Multimodal" and split == "test":
#         print(
#             "⚠️ Local evaluation for the test split of SWE-bench Multimodal is not supported. "
#             "Please check out sb-cli (https://github.com/swe-bench/sb-cli/) for instructions on how to submit predictions."
#         )
#         return

#     # set open file limit
#     if report_dir is not None:
#         report_dir = Path(report_dir)
#         if not report_dir.exists():
#             report_dir.mkdir(parents=True)

#     # load predictions as map of instance_id to prediction
#     # 这里是加载数据集的地方，这个路径是自己输入的
#     predictions = get_predictions_from_file(predictions_path, dataset_name, split)
#     predictions = {pred[KEY_INSTANCE_ID]: pred for pred in predictions}

#     # get dataset from predictions
#     dataset = get_dataset_from_preds(
#         dataset_name, split, instance_ids, predictions, run_id, rewrite_reports
#     )
#     full_dataset = load_swebench_dataset(dataset_name, split, instance_ids)


#     # run instances locally
#     if platform.system() == "Linux":
#         resource.setrlimit(resource.RLIMIT_NOFILE, (open_file_limit, open_file_limit))
#     client = docker.from_env()

#     existing_images = list_images(client)


#     # build environment images + run instances
#     if namespace is None and not rewrite_reports:
#         build_env_images(client, dataset, force_rebuild, max_workers)
#     run_instance(
#         predictions,
#         dataset,
#         cache_level,
#         clean,
#         force_rebuild,
#         max_workers,
#         run_id,
#         timeout,
#         namespace=namespace,
#         instance_image_tag=instance_image_tag,
#         rewrite_reports=rewrite_reports,
#     )

#     # clean images + make final report
#     clean_images(client, existing_images, cache_level, clean)
#     return make_run_report(predictions, full_dataset, run_id, client)

