import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from experiments import fill_missing as fm
from experiments import result_store as rs


class NotifierTest(unittest.TestCase):
    def setUp(self):
        self.saved = os.environ.pop("DISCORD_WEBHOOK_URL", None)

    def tearDown(self):
        os.environ.pop("DISCORD_WEBHOOK_URL", None)
        if self.saved is not None:
            os.environ["DISCORD_WEBHOOK_URL"] = self.saved

    def env_file(self, url):
        path = Path(tempfile.mkdtemp()) / ".env"
        path.write_text(f"DISCORD_WEBHOOK_URL={url}\n" if url else "")
        return path

    def test_missing_webhook_turns_notifications_off(self):
        with mock.patch.object(fm, "send_discord") as send, mock.patch.object(fm, "log") as log:
            notifier = fm.Notifier(self.env_file(""), "srv")
            self.assertFalse(notifier.send("hello"))
        send.assert_not_called()
        self.assertIn("off", log.call_args[0][0])

    def test_finished_tasks_are_batched_and_flushed_on_exit(self):
        with mock.patch.object(fm, "send_discord") as send:
            notifier = fm.Notifier(self.env_file("https://discord.com/api/webhooks/x/y"), "srv", batch_minutes=30)
            notifier.finished(["a--ETTh1--DLinear"])
            notifier.finished(["b--ETTh1--DLinear"])
            send.assert_not_called()                     # inside the batch window
            notifier.flush(force=True)
        self.assertEqual(send.call_count, 1)
        url, text = send.call_args[0]
        self.assertTrue(url.startswith("https://discord.com/api/webhooks/"))
        self.assertIn("2개", text)
        self.assertIn("[srv]", text)

    def test_failed_request_never_logs_the_webhook(self):
        url = "https://discord.com/api/webhooks/secret/token"
        with mock.patch.object(fm, "send_discord", side_effect=RuntimeError(f"boom {url}")), \
                mock.patch.object(fm, "log") as log:
            notifier = fm.Notifier(self.env_file(url), "srv")
            self.assertFalse(notifier.send("hello"))
        self.assertNotIn("secret", " ".join(str(c) for c in log.call_args_list))

    def test_progress_counts_electricity_and_traffic_apart(self):
        protocol, bases, plan = fm.load_context(fm.ROOT)
        tasks = rs.build_tasks(plan, protocol, bases, {})
        args = Namespace(phases="t3-none", datasets=None, backbones=None)
        text = fm.progress_text(tasks, args)
        # t3-none: 7 datasets x 2 backbones x 12 cells, nothing done
        self.assertIn("완료 0 runs", text)
        self.assertIn("Electricity 24", text)
        self.assertIn("Traffic 24", text)
        self.assertIn("그 외 120", text)
        # a worker limited to small datasets still reports the whole experiment
        small = Namespace(phases="t3-none", datasets="ETTh1,ETTm1,Weather", backbones=None)
        self.assertEqual(fm.progress_text(tasks, small), text)


if __name__ == "__main__":
    unittest.main()
