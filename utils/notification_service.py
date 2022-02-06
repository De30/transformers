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
import operator
import os
import re
import sys
import json

import requests
from slack_sdk import WebClient
from typing import Dict


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
        line = stacktrace[:stacktrace.index(' ')].split(':')[-2]
        error_message = stacktrace[stacktrace.index(' '):]

        stacktraces.append(f"(line {line}) {error_message}")

    return stacktraces


class Message:
    def __init__(self, title: str, model_results: Dict, additional_results: Dict):
        self.title = title

        # Failures and success of the modeling tests
        self.n_model_success = sum(r['success'] for r in model_results.values())
        self.n_model_failures = sum(r['failed']['total'] for r in model_results.values())

        # Failures and success of the additional tests
        self.n_additional_success = sum(r['success'] for r in additional_results.values())
        self.n_additional_failures = sum(r['failed'] for r in additional_results.values())

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
        # Create counters of categories for each model
        category_counter = [collections.Counter(v['failed']) for v in self.model_results.values()]

        # Sum all the counters
        category_sums = functools.reduce(operator.add, category_counter)

        # Remove the 'total', we're only interested in individual categories.
        del category_sums['total']

        category_failures_report = '\n'.join(sorted([f'*{k}*: {v}' for k, v in category_sums.items()]))

        return {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"The following modeling categories had failures:\n{category_failures_report}"
            }
        }

    @property
    def model_failures(self) -> Dict:
        # Obtain per-model failures
        failures = {k: v['failed']['total'] for k, v in self.model_results.items() if v['failed']['total']}

        model_failures_report = '\n'.join(sorted([f'*{k}*: {v}' for k, v in failures.items()]))

        return {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"These following model modules had failures:\n{model_failures_report}"
            }
        }

    @property
    def additional_failures(self) -> Dict:
        failures = '\n'.join(sorted([f'*{k}*: {v["failed"]}' for k, v in self.additional_results.items() if v['failed'] > 0]))

        return {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"The following non-modeling tests had failures:\n{failures}"
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

    def post(self):
        print("Sending the following payload")
        print(json.dumps({"blocks": json.loads(self.payload)}))
        self.thread_ts = client.chat_postMessage(
            channel=os.environ["CI_SLACK_CHANNEL_DUMMY_TESTS"],
            blocks=self.payload,
            text=f"{self.n_failures} failures out of {self.n_tests} tests," if self.n_failures else "All tests passed."
        )

    def get_reply_blocks(self, job_name, job_result, text):
        failures = job_result['failures']

        if len(failures) > 2500:
            failures = '\n'.join(job_result['failures'].split('\n')[:20]) + '\n\n[Truncated]'

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
                    "text": job_name.upper(),
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
                del job_result['failed']['total']

                failures = '\n'.join(sorted([f'*{k}*: {v}' for k, v in job_result['failed'].items() if v > 0]))

                blocks = self.get_reply_blocks(job, job_result, text=failures)

                print("Sending the following reply")
                print(json.dumps({"blocks": blocks}))

                client.chat_postMessage(
                    channel=os.environ["CI_SLACK_CHANNEL_DUMMY_TESTS"],
                    text=f"Results for {job}",
                    blocks=blocks,
                    thread_ts=self.thread_ts["ts"]
                )

        for job, job_result in self.additional_results.items():
            if len(job_result["failures"]):
                blocks = self.get_reply_blocks(job, job_result, text=f"Number of failures: {job_result['failed']}")

                print("Sending the following reply")
                print(json.dumps({"blocks": blocks}))

                client.chat_postMessage(
                    channel=os.environ["CI_SLACK_CHANNEL_DUMMY_TESTS"],
                    text=f"Results for {job}",
                    blocks=blocks,
                    thread_ts=self.thread_ts["ts"]
                )


def get_job_links():
    run_id = os.environ['GITHUB_RUN_ID']
    result = requests.get(f"https://api.github.com/repos/huggingface/transformers/actions/runs/{run_id}/jobs").json()

    try:
        return {job['name']: job['html_url'] for job in result['jobs']}
    except Exception as e:
        print("Unknown error, could not fetch links.", e)

    return {}


if __name__ == "__main__":
    arguments = sys.argv[1:][0]
    models = ast.literal_eval(arguments)

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

    # This dict will contain all the information relative to each model:
    # - Failures: the total, as well as the number of failures per-category defined above
    # - Success: total
    # - Time spent: as a comma-separated list of elapsed time
    # - Failures: as a line-break separated list of errors
    model_results = {
        model: {
            "failed": {'total': 0, **{m: 0 for m in modeling_categories}},
            "success": 0,
            "time_spent": "",
            "failures": ""
        } for model in models if os.path.exists(f'run_all_tests_gpu_{model}_test_reports')
    }

    unclassified_model_failures = []

    for model in models:
        if os.path.exists(f'run_all_tests_gpu_{model}_test_reports'):
            # Link to the GitHub Action job
            model_results[model]['job_link'] = github_actions_job_links.get(f"Model tests ({model})")

            files = os.listdir(f'run_all_tests_gpu_{model}_test_reports')

            with open(os.path.join(f"run_all_tests_gpu_{model}_test_reports", "stats.txt")) as f:
                failed, success, time_spent = handle_test_results(f.read())
                model_results[model]["failed"]['total'] += failed
                model_results[model]["success"] += success
                model_results[model]["time_spent"] += time_spent[1:-1] + ", "

            with open(os.path.join(f"run_all_tests_gpu_{model}_test_reports", "failures_line.txt")) as failures_line:
                total_failure_lines = failures_line.read()

            stacktraces = handle_stacktraces(total_failure_lines)

            with open(os.path.join(f"run_all_tests_gpu_{model}_test_reports", "summary_short.txt")) as summary_short:
                for line in summary_short:
                    if re.search("FAILED", line):

                        line = line.replace("FAILED ", "")
                        line = line.split()[0].replace('\n', '')
                        model_results[model]["failures"] += f"*{line}*\n_{stacktraces.pop(0)}_\n\n"

                        if re.search("_tf_", line):
                            model_results[model]['failed']["TensorFlow"] += 1

                        elif re.search("_flax_", line):
                            model_results[model]['failed']['Flax'] += 1

                        elif re.search("test_modeling", line):
                            model_results[model]['failed']['PyTorch'] += 1

                        elif re.search('test_tokenization', line):
                            model_results[model]['failed']['Tokenizers'] += 1

                        elif re.search('test_pipelines', line):
                            model_results[model]['failed']['Pipelines'] += 1

                        elif re.search('test_trainer', line):
                            model_results[model]['failed']['Trainer'] += 1

                        elif re.search('onnx', line):
                            model_results[model]['failed']['ONNX'] += 1

                        elif re.search('auto', line):
                            model_results[model]['failed']['Auto'] += 1

                        else:
                            model_results[model]['failed']['Unclassified'] += 1
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
            "failed": 0,
            "success": 0,
            "time_spent": "",
            "failures": "",
            "job_link": github_actions_job_links.get(key)
        } for key in additional_files.keys()
    }

    for key in additional_results.keys():
        with open(os.path.join(additional_files[key], "failures_line.txt")) as failures_line:
            total_failure_lines = failures_line.read()

        stacktraces = handle_stacktraces(total_failure_lines)

        with open(os.path.join(additional_files[key], 'stats.txt')) as stats:
            failed, success, time_spent = handle_test_results(stats.read())
            additional_results[key]["failed"] += failed
            additional_results[key]["success"] += success
            additional_results[key]["time_spent"] += time_spent[1:-1] + ", "

            if failed:
                with open(os.path.join(additional_files[key], 'summary_short.txt')) as summary_short:
                    for line in summary_short:
                        if re.search("FAILED", line):
                            line = line.replace("FAILED ", "")
                            line = line.split()[0].replace('\n', '')
                            additional_results[key]["failures"] += f"*{line}*\n_{stacktraces.pop(0)}_\n\n"

    message = Message("🤗 Results of the scheduled tests.", model_results, additional_results)

    message.post()
    message.post_reply()
