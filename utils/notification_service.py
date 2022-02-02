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
import os
import re
import sys
import json

from slack_sdk import WebClient
from typing import Dict, Tuple


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


class Message:
    def __init__(self, title: str, module_failures: Dict, category_failures: Dict, results: Dict):
        self.title = title

        self._category_failures = category_failures
        self._module_failures = module_failures

        self.n_failures = sum(category_failures.values())
        self.n_tests = sum(r['success'] + r['failed'] for r in results.values())
        self.results = results

        self.thread_ts = None

    @property
    def time(self) -> str:
        time_spent = [r['time_spent'].split(', ')[0] for r in self.results.values()]
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
                "value": "click_me_123",
                "url": f"https://github.com/huggingface/transformers/actions/runs/{os.environ['GITHUB_RUN_ID']}",
                "action_id": "button-action"
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
                "value": "click_me_123",
                "url": f"https://github.com/huggingface/transformers/actions/runs/{os.environ['GITHUB_RUN_ID']}",
                "action_id": "button-action"
            }
        }

    @property
    def category_failures(self) -> Dict:
        failures = '\n'.join(sorted([f'*{k}*: {v}' for k, v in self._category_failures.items()]))

        return {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"The following categories had failures:\n{failures}"
            }
        }

    @property
    def module_failures(self) -> Dict:
        failures = '\n'.join(sorted([f'*{k}*: {v}' for k, v in self._module_failures.items()]))

        return {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"These following modules had failures:\n{failures}"
            }
        }

    @property
    def payload(self) -> str:
        blocks = [self.header]

        if self.n_failures > 0:
            blocks.extend([self.failures, self.category_failures, self.module_failures])
        else:
            blocks.append(self.no_failures)

        return json.dumps({"blocks": blocks})


    def post(self):
        self.thread_ts = client.chat_postMessage(
            channel=os.environ["CI_SLACK_CHANNEL_DUMMY_TESTS"],
            blocks=self.payload,
        )

    def post_reply(self):
        if self.thread_ts is None:
            raise ValueError("Can only post reply if a post has been made.")

        for job, job_result in results.items():
            if len(job_result["failures"]):
                client.chat_postMessage(
                    channel=os.environ["CI_SLACK_CHANNEL_DUMMY_TESTS"],
                    text=f"{job}\n{job_result['failures']}",
                    thread_ts=self.thread_ts
                )


if __name__ == "__main__":
    arguments = sys.argv[1:][0]
    models = ast.literal_eval(arguments)

    if len(models) == 0:
        models = [a.split("/")[-1] for a in list(filter(os.path.isdir, (os.path.join("/home/lysandre/transformers/tests", f) for f in os.listdir("/home/lysandre/transformers/tests"))))]

    results = {}
    failure_categories = {
        "PyTorch": 0,
        "TensorFlow": 0,
        "Flax": 0,
        "Tokenizers": 0,
        "Pipelines": 0,
        "Trainer": 0,
        "ONNX": 0,
        "Unclassified": 0
    }
    module_failures = {}
    unclassified_failures = []
    for model in models:
        if os.path.exists(f'run_all_tests_gpu_{model}_test_reports'):
            results[model] = {"failed": 0, "success": 0, "time_spent": "", "failures": ""}
            files = os.listdir(f'run_all_tests_gpu_{model}_test_reports')

            with open(os.path.join(f"run_all_tests_gpu_{model}_test_reports", f"tests_gpu_{model}_stats.txt")) as f:
                failed, success, time_spent = handle_test_results(f.read())

                if failed:
                    module_failures[model] = failed

                results[model]["failed"] += failed
                results[model]["success"] += success
                results[model]["time_spent"] += time_spent[1:-1] + ", "

            with open(os.path.join(f"run_all_tests_gpu_{model}_test_reports", f"tests_gpu_{model}_summary_short.txt")) as f:
                for line in f:
                    if re.search("FAILED", line):
                        results[model]["failures"] += line

                        if re.search('test_modeling', line):
                            if re.search("_tf_", line):
                                failure_categories['TensorFlow'] += 1
                            elif re.search("_flax_", line):
                                failure_categories['Flax'] += 1
                            else:
                                failure_categories['PyTorch'] += 1

                        elif re.search('test_tokenization', line):
                            failure_categories['Tokenizers'] += 1

                        elif re.search('test_pipelines', line):
                            failure_categories['Pipelines'] += 1

                        elif re.search('test_trainer', line):
                            failure_categories['Trainer'] += 1

                        elif re.search('onnx', line):
                            failure_categories['ONNX'] += 1

                        else:
                            failure_categories['Unclassified'] += 1
                            unclassified_failures.append(line)
    
    message = Message(
        title="🤗 Results of the scheduled tests.",
        module_failures=module_failures,
        category_failures={k: v for k, v in failure_categories.items() if v > 0},
        results=results
    )

    print(message.payload)

    message.post()
    message.post_reply()