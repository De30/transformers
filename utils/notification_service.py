# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import ast
import collections
import functools
import math
import operator
import os
import re
import sys
import json
import time

import requests
from slack_sdk import WebClient
from typing import Dict, Optional, List, Union

client = WebClient(token=os.environ["CI_SLACK_BOT_TOKEN"])


def handle_test_results(test_results):
    expressions = test_results.split(" ")

    failed = 0
    success = 0

    # When the output is short enough, the output is surrounded by = signs: "== OUTPUT =="
    # When it is too long, those signs are not present.
    time_spent = expressions[-2] if "=" in expressions[-1] else expressions[-1]

    for i, expression in enumerate(expressions):
        if "failed" in expression:
            failed += int(expressions[i - 1])
        if "passed" in expression:
            success += int(expressions[i - 1])

    return failed, success, time_spent


def handle_stacktraces(test_results):
    # These files should follow the following architecture:
    # === FAILURES ===
    # <path>:<line>: Error ...
    # <path>:<line>: Error ...
    # <empty line>

    total_stacktraces = test_results.split('\n')[1:-1]
    stacktraces = []
    for stacktrace in total_stacktraces:
        try:
            line = stacktrace[:stacktrace.index(' ')].split(':')[-2]
            error_message = stacktrace[stacktrace.index(' '):]

            stacktraces.append(f"(line {line}) {error_message}")
        except Exception:
            stacktraces.append("Cannot retrieve error message.")


    return stacktraces


def dicts_to_sum(objects: Union[Dict[str, Dict], List[dict]]):
    if isinstance(objects, dict):
        lists = objects.values()
    else:
        lists = objects

    # Convert each dictionary to counter
    counters = map(collections.Counter, lists)
    # Sum all the counters
    return functools.reduce(operator.add, counters)


class Message:
    def __init__(self, title: str, model_results: Dict, additional_results: Dict):
        self.title = title

        # Failures and success of the modeling tests
        self.n_model_success = sum(r['success'] for r in model_results.values())
        self.n_model_single_gpu_failures = sum(dicts_to_sum(r['failed'])['single'] for r in model_results.values())
        self.n_model_multi_gpu_failures = sum(dicts_to_sum(r['failed'])['multi'] for r in model_results.values())

        # Some suites do not have a distinction between single and multi GPU.
        self.n_model_unknown_failures = sum(dicts_to_sum(r['failed'])['unclassified'] for r in model_results.values())
        self.n_model_failures = self.n_model_single_gpu_failures + self.n_model_multi_gpu_failures + self.n_model_unknown_failures

        # Failures and success of the additional tests
        self.n_additional_success = sum(r['success'] for r in additional_results.values())

        all_additional_failures = dicts_to_sum([r['failed'] for r in additional_results.values()])
        self.n_additional_single_gpu_failures = all_additional_failures['single']
        self.n_additional_multi_gpu_failures = all_additional_failures['multi']
        self.n_additional_unknown_gpu_failures = all_additional_failures['unclassified']
        self.n_additional_failures = self.n_additional_single_gpu_failures + self.n_additional_multi_gpu_failures + self.n_additional_unknown_gpu_failures

        # Results
        self.n_failures = self.n_model_failures + self.n_additional_failures
        self.n_success = self.n_model_success + self.n_additional_success
        self.n_tests = self.n_failures + self.n_success

        self.model_results = model_results
        self.additional_results = additional_results

        self.thread_ts = None

    @property
    def time(self) -> str:
        time_spent = [r['time_spent'].split(', ')[0] for r in [*self.model_results.values(), *self.additional_results.values()]]
        total_secs = 0

        for time in time_spent:
            time_parts = time.split(':')

            # Time can be formatted as xx:xx:xx, as .xx, or as x.xx if the time spent was less than a minute.
            if len(time_parts) == 1:
                time_parts = [0, 0, time_parts[0]]

            hours, minutes, seconds = int(time_parts[0]), int(time_parts[1]), float(time_parts[2])
            total_secs += hours * 3600 + minutes * 60 + seconds

        hours, minutes, seconds = total_secs // 3600, (total_secs % 3600) // 60, total_secs % 60
        return f"{int(hours)}h{int(minutes)}m{int(seconds)}s"

    @property
    def header(self) -> Dict:
        return {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": self.title
            }
        }

    @property
    def no_failures(self) -> Dict:
        return {
            "type": "section",
            "text": {
                "type": "plain_text",
                "text": f"🌞 There were no failures: all {self.n_tests} tests passed. The suite ran in {self.time}.",
                "emoji": True
            },
            "accessory": {
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": "Check Action results",
                    "emoji": True
                },
                "url": f"https://github.com/huggingface/transformers/actions/runs/{os.environ['GITHUB_RUN_ID']}",
            }
        }

    @property
    def failures(self) -> Dict:
        return {
            "type": "section",
            "text": {
                "type": "plain_text",
                "text": f"There were {self.n_failures} failures, out of {self.n_tests} tests.\nThe suite ran in {self.time}.",
                "emoji": True
            },
            "accessory": {
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": "Check Action results",
                    "emoji": True
                },
                "url": f"https://github.com/huggingface/transformers/actions/runs/{os.environ['GITHUB_RUN_ID']}",
            }
        }

    @property
    def category_failures(self) -> Dict:
        model_failures = [v['failed'] for v in self.model_results.values()]

        category_failures = {}

        for model_failure in model_failures:
            for key, value in model_failure.items():
                if key not in category_failures:
                    category_failures[key] = value
                else:
                    category_failures[key]['unclassified'] += value['unclassified']
                    category_failures[key]['single'] += value['single']
                    category_failures[key]['multi'] += value['multi']

        individual_reports = []
        for key, value in category_failures.items():
            if 'single' in value and 'multi' in value:
                device_report = f"{value['single']}/{value['multi']}"
            elif 'single' in value:
                device_report = f"{value['single']}/0"
            elif 'multi' in value:
                device_report = f"0/{value['multi']}"
            else:
                device_report = None

            if sum(value.values()):
                report = f"*{key}*: {sum(value.values())}"
                if device_report:
                    report = f"{report} [{device_report}]"

                individual_reports.append(report)

        category_failures_report = '\n'.join(sorted(individual_reports))

        return {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"The following modeling categories had failures [single/multi]:\n{category_failures_report}"
            }
        }

    @property
    def model_failures(self) -> Dict:
        # Obtain per-model failures
        def per_model_sum(model_category_dict):
            return dicts_to_sum(model_category_dict['failed'].values())

        failures = {k: per_model_sum(v) for k, v in self.model_results.items() if sum(per_model_sum(v).values())}

        individual_reports = []
        for key, value in failures.items():
            if 'single' in value and 'multi' in value:
                device_report = f"{value['single']}/{value['multi']}"
            elif 'single' in value:
                device_report = f"{value['single']}/0"
            elif 'multi' in value:
                device_report = f"0/{value['multi']}"
            else:
                device_report = None

            if sum(value.values()):
                report = f"*{key}*: {sum(value.values())}"
                if device_report:
                    report = f"{report} [{device_report}]"

                individual_reports.append(report)

        model_failures_report = '\n'.join(sorted(individual_reports))

        return {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"These following model modules had failures [single/multi]:\n{model_failures_report}"
            }
        }

    @property
    def additional_failures(self) -> Dict:
        failures = {k: v['failed'] for k, v in self.additional_results.items() if sum(v['failed'].values())}

        individual_reports = []
        for key, value in failures.items():
            if 'single' in value and 'multi' in value:
                device_report = f"{value['single']}/{value['multi']}"
            elif 'single' in value:
                device_report = f"{value['single']}/0"
            elif 'multi' in value:
                device_report = f"0/{value['multi']}"
            else:
                device_report = None

            if sum(value.values()):
                report = f"*{key}*: {sum(value.values())}"
                if device_report:
                    report = f"{report} [{device_report}]"

                individual_reports.append(report)

        failures_report = '\n'.join(sorted(individual_reports))

        return {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"The following non-modeling tests had failures [single/multi]:\n{failures_report}"
            }
        }

    @property
    def payload(self) -> str:
        blocks = [self.header]

        if self.n_model_failures > 0 or self.n_additional_failures > 0:
            blocks.append(self.failures)

        if self.n_model_failures > 0:
            blocks.extend([self.category_failures, self.model_failures])

        if self.n_additional_failures > 0:
            blocks.append(self.additional_failures)

        if self.n_model_failures == 0 and self.n_additional_failures == 0:
            blocks.append(self.no_failures)

        return json.dumps(blocks)

    @staticmethod
    def error_out():
        client.chat_postMessage(
            channel=os.environ["CI_SLACK_CHANNEL_DUMMY_TESTS"],
            text="There was an issue running the tests."
        )


    def post(self):
        print("Sending the following payload")
        print(json.dumps({"blocks": json.loads(self.payload)}))
        self.thread_ts = client.chat_postMessage(
            channel=os.environ["CI_SLACK_CHANNEL_DUMMY_TESTS"],
            blocks=self.payload,
            text=f"{self.n_failures} failures out of {self.n_tests} tests," if self.n_failures else "All tests passed."
        )

    def get_reply_blocks(self, job_name, job_result, failures, device, text):
        if len(failures) > 2500:
            failures = '\n'.join(failures.split('\n')[:20]) + '\n\n[Truncated]'

        title = job_name
        if device is not None:
            title += f" ({device}-gpu)"

        content = {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": text
            },
        }

        if job_result['job_link'] is not None:
            content['accessory'] = {
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": "GitHub Action job",
                    "emoji": True
                },
                "url": job_result['job_link'],
            }

        return [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": title.upper(),
                    "emoji": True
                }
            },
            content,
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": failures
                }
            }
        ]

    def post_reply(self):
        if self.thread_ts is None:
            raise ValueError("Can only post reply if a post has been made.")

        for job, job_result in self.model_results.items():
            if len(job_result["failures"]):
                for device, failures in job_result['failures'].items():
                    text = '\n'.join(sorted([f'*{k}*: {sum(v.values())}' for k, v in job_result['failed'].items() if sum(v.values())]))

                    blocks = self.get_reply_blocks(job, job_result, failures, device, text=text)

                    print("Sending the following reply")
                    print(json.dumps({"blocks": blocks}))

                    client.chat_postMessage(
                        channel=os.environ["CI_SLACK_CHANNEL_DUMMY_TESTS"],
                        text=f"Results for {job}",
                        blocks=blocks,
                        thread_ts=self.thread_ts["ts"]
                    )

                    time.sleep(1)

        for job, job_result in self.additional_results.items():
            if len(job_result["failures"]):
                for device, failures in job_result['failures'].items():
                    blocks = self.get_reply_blocks(job, job_result, failures, device, text=f"Number of failures: {sum(job_result['failed'].values())}")

                    print("Sending the following reply")
                    print(json.dumps({"blocks": blocks}))

                    client.chat_postMessage(
                        channel=os.environ["CI_SLACK_CHANNEL_DUMMY_TESTS"],
                        text=f"Results for {job}",
                        blocks=blocks,
                        thread_ts=self.thread_ts["ts"]
                    )

                    time.sleep(1)


def get_job_links():
    run_id = os.environ['GITHUB_RUN_ID']
    url = f"https://api.github.com/repos/huggingface/transformers/actions/runs/{run_id}/jobs?per_page=100"
    result = requests.get(url).json()
    jobs = {}

    try:
        jobs.update({job['name']: job['html_url'] for job in result['jobs']})
        pages_to_iterate_over = math.ceil((result['total_count'] - 100) / 100)

        for i in range(pages_to_iterate_over):
            result = requests.get(url + f"&page={i + 2}").json()
            jobs.update({job['name']: job['html_url'] for job in result['jobs']})

        return jobs
    except Exception as e:
        print("Unknown error, could not fetch links.", e)

    return {}


def retrieve_artifact(name: str, gpu: Optional[str]):
    if gpu not in [None, 'single', 'multi']:
        raise ValueError(f'Invalid GPU for artifact. Passed GPU: `{gpu}`.')

    if gpu is not None:
        name = f"{gpu}-gpu-docker_{name}"

    _artifact = {}

    if os.path.exists(name):
        files = os.listdir(name)
        for file in files:
            try:
                with open(os.path.join(name, file)) as f:
                    _artifact[file.split('.')[0]] = f.read()
            except UnicodeDecodeError as e:
                raise ValueError(f"Could not open {os.path.join(name, file)}.") from e

    return _artifact


def retrieve_available_artifacts():
    class Artifact:
        def __init__(self, name: str, single_gpu: bool = False, multi_gpu: bool = False):
            self.name = name
            self.single_gpu = single_gpu
            self.multi_gpu = multi_gpu
            self.paths = []

        def __str__(self):
            return self.name

        def add_path(self, path: str, gpu: str = None):
            self.paths.append({'name': self.name, 'path': path, 'gpu': gpu})

    _available_artifacts: Dict[str, Artifact] = {}

    directories = filter(os.path.isdir, os.listdir())
    for directory in directories:
        if directory.startswith('single-gpu-docker'):
            artifact_name = directory[len('single-gpu-docker') + 1:]

            if artifact_name in _available_artifacts:
                _available_artifacts[artifact_name].single_gpu = True
            else:
                _available_artifacts[artifact_name] = Artifact(artifact_name, single_gpu=True)

            _available_artifacts[artifact_name].add_path(directory, gpu='single')

        elif directory.startswith('multi-gpu-docker'):
            artifact_name = directory[len('multi-gpu-docker') + 1:]

            if artifact_name in _available_artifacts:
                _available_artifacts[artifact_name].multi_gpu = True
            else:
                _available_artifacts[artifact_name] = Artifact(artifact_name, multi_gpu=True)

            _available_artifacts[artifact_name].add_path(directory, gpu='multi')
        else:
            artifact_name = directory
            if artifact_name not in _available_artifacts:
                _available_artifacts[artifact_name] = Artifact(artifact_name)

            _available_artifacts[artifact_name].add_path(directory)

    return _available_artifacts


if __name__ == "__main__":
    arguments = sys.argv[1:][0]
    try:
        models = ast.literal_eval(arguments)
    except SyntaxError:
        Message.error_out()


    if len(models) == 0:
        models = [a.split("/")[-1] for a in list(filter(os.path.isdir, (os.path.join("/home/lysandre/transformers/tests", f) for f in os.listdir("/home/lysandre/transformers/tests"))))]

    github_actions_job_links = get_job_links()

    modeling_categories = [
        "PyTorch",
        "TensorFlow",
        "Flax",
        "Tokenizers",
        "Pipelines",
        "Trainer",
        "ONNX",
        "Auto",
        "Unclassified"
    ]

    available_artifacts = retrieve_available_artifacts()
    # This dict will contain all the information relative to each model:
    # - Failures: the total, as well as the number of failures per-category defined above
    # - Success: total
    # - Time spent: as a comma-separated list of elapsed time
    # - Failures: as a line-break separated list of errors
    model_results = {
        model: {
            "failed": {m: {'unclassified': 0, 'single': 0, 'multi': 0} for m in modeling_categories},
            "success": 0,
            "time_spent": "",
            "failures": {}
        } for model in models if f'run_all_tests_gpu_{model}_test_reports' in available_artifacts
    }

    unclassified_model_failures = []

    for model in model_results.keys():
        for artifact_path in available_artifacts[f'run_all_tests_gpu_{model}_test_reports'].paths:
            artifact = retrieve_artifact(artifact_path['name'], artifact_path['gpu'])
            if 'stats' in artifact:
                # Link to the GitHub Action job
                model_results[model]['job_link'] = github_actions_job_links.get(f"Model tests ({model}, {artifact_path['gpu']}-gpu-docker)")

                failed, success, time_spent = handle_test_results(artifact['stats'])
                model_results[model]["success"] += success
                model_results[model]["time_spent"] += time_spent[1:-1] + ", "

                stacktraces = handle_stacktraces(artifact['failures_line'])

                for line in artifact['summary_short'].split('\n'):
                    if re.search("FAILED", line):

                        line = line.replace("FAILED ", "")
                        line = line.split()[0].replace('\n', '')

                        if artifact_path['gpu'] not in model_results[model]["failures"]:
                            model_results[model]["failures"][artifact_path['gpu']] = ""

                        model_results[model]["failures"][artifact_path['gpu']] += f"*{line}*\n_{stacktraces.pop(0)}_\n\n"

                        if re.search("_tf_", line):
                            model_results[model]['failed']["TensorFlow"][artifact_path['gpu']] += 1

                        elif re.search("_flax_", line):
                            model_results[model]['failed']['Flax'][artifact_path['gpu']] += 1

                        elif re.search("test_modeling", line):
                            model_results[model]['failed']['PyTorch'][artifact_path['gpu']] += 1

                        elif re.search('test_tokenization', line):
                            model_results[model]['failed']['Tokenizers'][artifact_path['gpu']] += 1

                        elif re.search('test_pipelines', line):
                            model_results[model]['failed']['Pipelines'][artifact_path['gpu']] += 1

                        elif re.search('test_trainer', line):
                            model_results[model]['failed']['Trainer'][artifact_path['gpu']] += 1

                        elif re.search('onnx', line):
                            model_results[model]['failed']['ONNX'][artifact_path['gpu']] += 1

                        elif re.search('auto', line):
                            model_results[model]['failed']['Auto'][artifact_path['gpu']] += 1

                        else:
                            model_results[model]['failed']['Unclassified'][artifact_path['gpu']] += 1
                            unclassified_model_failures.append(line)

    # Additional runs
    additional_files = {
        'Examples directory': "run_examples_gpu",
        'PyTorch pipelines': "run_tests_torch_pipeline_gpu",
        'TensorFlow pipelines': "run_tests_tf_pipeline_gpu",
        'Torch CUDA extension tests': "run_tests_torch_cuda_extensions_gpu_test_reports"
    }

    additional_results = {
        key: {
            "failed": {'unclassified': 0, 'single': 0, 'multi': 0},
            "success": 0,
            "time_spent": "",
            "failures": {},
            "job_link": github_actions_job_links.get(key)
        } for key in additional_files.keys()
    }

    for key in additional_results.keys():
        for artifact_path in available_artifacts[additional_files[key]].paths:
            if artifact_path['gpu'] is not None:
                additional_results[key]['job_link'] = github_actions_job_links.get(f"{key} ({artifact_path['gpu']}-gpu-docker)")
            artifact = retrieve_artifact(artifact_path['name'], artifact_path['gpu'])
            stacktraces = handle_stacktraces(artifact['failures_line'])

            failed, success, time_spent = handle_test_results(artifact['stats'])
            additional_results[key]["failed"][artifact_path['gpu'] or 'unclassified'] += failed
            additional_results[key]["success"] += success
            additional_results[key]["time_spent"] += time_spent[1:-1] + ", "

            if failed:
                for line in artifact['summary_short'].split('\n'):
                    if re.search("FAILED", line):
                        line = line.replace("FAILED ", "")
                        line = line.split()[0].replace('\n', '')

                        if artifact_path['gpu'] not in additional_results[key]['failures']:
                            additional_results[key]['failures'][artifact_path['gpu']] = ""

                        additional_results[key]["failures"][artifact_path['gpu']] += f"*{line}*\n_{stacktraces.pop(0)}_\n\n"

    message = Message("🤗 Results of the scheduled tests.", model_results, additional_results)

    message.post()
    message.post_reply()
