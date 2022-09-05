import json
import os
from slack_sdk import WebClient


client = WebClient(token=os.environ["CI_SLACK_BOT_TOKEN"])


class Message:

    @property
    def payload(self) -> str:
        blocks = [{"type": "header", "text": {"type": "plain_text", "text": "🤗 Results of the Past CI - pytorch-1.11 tests."}}, {"type": "section", "text": {"type": "plain_text", "text": "There were 68 failures, out of 28636 tests.\nThe suite ran in 2h34m15s.", "emoji": True}, "accessory": {"type": "button", "text": {"type": "plain_text", "text": "Check Action results", "emoji": True}, "url": "https://github.com/huggingface/transformers/actions/runs/2970011128"}}, {"type": "section", "text": {"type": "mrkdwn", "text": "The following modeling categories had failures:\n```\nSingle |  Multi | Category\n    14 |     15 | PyTorch\n     3 |      3 | TensorFlow\n     1 |      1 | Tokenizers\n     1 |      0 | Trainer\n     1 |      1 | Auto\n    14 |     14 | Unclassified\n```\n"}}, {"type": "section", "text": {"type": "mrkdwn", "text": "These following model modules had failures:\n```\nSingle PT |  Multi PT | Single TF |  Multi TF |     Other | Category\n        0 |         0 |         0 |         0 |         1 | trainer\n        0 |         0 |         0 |         0 |         2 | models_layoutlmv2\n        0 |         0 |         0 |         0 |        30 | models_wav2vec2_with_lm\n        0 |         0 |         3 |         3 |         0 | models_opt\n        1 |         1 |         0 |         0 |         0 | models_wav2vec2\n        3 |         3 |         0 |         0 |         0 | models_bloom\n       10 |        11 |         0 |         0 |         0 | models_owlvit\n```\n"}}, {"type": "section", "text": {"type": "mrkdwn", "text": "The following non-model modules had failures:\n```\nSingle |  Multi | Category\n     1 |      0 | trainer\n```\n"}}]
        return json.dumps(blocks)

    def post(self):
        self.thread_ts = client.chat_postMessage(
            channel=os.environ["CI_SLACK_REPORT_CHANNEL_ID"],
            blocks=self.payload,
            text="dummy text",
        )


if __name__ == "__main__":

    message = Message()
    message.post()
