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

import os
import re
import sys

from slack_sdk import WebClient


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


def format_for_slack(total_results, results, scheduled: bool, title: str):
    print(total_results, results)
    header = {
        "type": "header",
        "text": {
            "type": "plain_text",
            "text": title,
            "emoji": True,
        },
    }

    if total_results["failed"] > 0:
        total = {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Failures:*\n❌ {total_results['failed']} failures."},
                {"type": "mrkdwn", "text": f"*Passed:*\n✅ {total_results['success']} tests passed."},
            ],
        }
    else:
        total = {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": "\n🌞 All tests passed."},
            ],
        }

    blocks = [header, total]

    if total_results["failed"] > 0:
        for key, result in results.items():
            print(key, result)
            blocks.append({"type": "header", "text": {"type": "plain_text", "text": key, "emoji": True}})
            blocks.append(
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Results:*\n{result['failed']} failed, {result['success']} passed.",
                        },
                        {"type": "mrkdwn", "text": f"*Time spent:*\n{result['time_spent']}"},
                    ],
                }
            )
    elif not scheduled:
        for key, result in results.items():
            blocks.append(
                {"type": "section", "fields": [{"type": "mrkdwn", "text": f"*{key}*\n{result['time_spent']}."}]}
            )

    footer = {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": f"<https://github.com/huggingface/transformers/actions/runs/{os.environ['GITHUB_RUN_ID']}|View on GitHub>",
        },
    }

    blocks.append(footer)

    blocks = {"blocks": blocks}

    return blocks


if __name__ == "__main__":
    arguments = sys.argv[1:]

    print(arguments)