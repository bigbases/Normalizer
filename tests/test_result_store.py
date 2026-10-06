import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from experiments import result_store as rs
from experiments.run_matrix import BASE_CONFIGS_PATH, PROTOCOL_PATH, load_json


def fake_rows(cells, value=0.5):
    rows = {}
    for i, c in enumerate(cells):
        rows[c["run_id"]] = dict(
            RunID=c["run_id"], CandidateID=c["candidate_id"], ConfigHash=c["config_hash"],
            Phase=c["stage"], Split="test" if c["stage"] == "final" else "validation",
            Dataset=c["dataset"], Backbone=c["backbone"], Horizon=str(c["horizon"]),
            Seed=str(c["seed"]), BestValMSE=str(value + i * 1e-3), UseNorm=c["method"],
            MSE="0.3", MAE="0.4",
        )
    return rows


class TaskResolutionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_json(PROTOCOL_PATH)
        cls.bases = load_json(BASE_CONFIGS_PATH)
        cls.plan = rs.load_plan()

    def tasks(self, results):
        tasks = rs.build_tasks(self.plan, self.protocol, self.bases, results)
        return {t.task_id: t for t in tasks}

    def test_dependencies_unblock_from_packaged_results(self):
        t = self.tasks({})
        self.assertEqual(t["exp1-lt--ETTh1--DLinear"].status, "ready")
        self.assertEqual(len(t["exp1-lt--ETTh1--DLinear"].cells), 12)
        self.assertEqual(len(t["exp2-search--ETTh1--DLinear"].cells), 40)
        self.assertEqual(t["exp2-confirm--ETTh1--DLinear"].status, "blocked")
        self.assertEqual(t["exp3-tuned--ETTh1--DLinear"].status, "blocked")

        results = fake_rows(t["exp2-search--ETTh1--DLinear"].cells)
        t = self.tasks(results)
        self.assertEqual(t["exp2-search--ETTh1--DLinear"].status, "done")
        confirm = t["exp2-confirm--ETTh1--DLinear"]
        self.assertEqual(confirm.status, "ready")
        self.assertEqual(len(confirm.cells), 4 * 2 * 2)       # methods x top-2 x horizons
        self.assertEqual(t["exp2-confirm--ETTh2--DLinear"].status, "blocked")

        results.update(fake_rows(confirm.cells))
        tuned = self.tasks(results)["exp3-tuned--ETTh1--DLinear"]
        self.assertEqual(tuned.status, "ready")
        self.assertEqual(len(tuned.cells), 4 * 4 * 3)         # methods x horizons x seeds

    def test_claim_status_lease_and_failure(self):
        task_id = "exp1-lt--ETTh1--DLinear"
        now = 1_000_000_000.0
        fresh = {"worker": "w", "heartbeat": "2001-09-09T01:46:40+00:00"}      # == now
        stale = {"worker": "w", "heartbeat": "2001-09-08T20:46:40+00:00"}      # now - 5 h
        failed = {**fresh, "status": "failed"}
        for claim, expected in ((fresh, "claimed"), (stale, "ready"), (failed, "failed")):
            tasks = rs.build_tasks(self.plan, self.protocol, self.bases, {}, {task_id: claim},
                                   lease_hours=3, now=now)
            status = next(t.status for t in tasks if t.task_id == task_id)
            self.assertEqual(status, expected)

    def test_write_cell_is_idempotent_per_run_id(self):
        root = Path(tempfile.mkdtemp())
        try:
            cell = self.tasks({})["exp1-lt--ETTh1--DLinear"].cells[0]
            row = next(iter(fake_rows([cell]).values()))
            self.assertTrue(rs.write_cell(root, row))
            self.assertFalse(rs.write_cell(root, {**row, "MSE": "9.9"}))
            self.assertEqual(rs.read_store(root)[cell["run_id"]]["MSE"], "0.3")
        finally:
            shutil.rmtree(root)


def git(cwd, *args):
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True)


class GitTransactionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.remote = self.tmp / "remote.git"
        git(self.tmp, "init", "-q", "--bare", "-b", "main", str(self.remote))
        seed = self.tmp / "seed"
        git(self.tmp, "clone", "-q", str(self.remote), str(seed))
        (seed / "code.py").write_text("x = 1\n")
        for cmd in (("config", "user.email", "t@t"), ("config", "user.name", "t"),
                    ("add", "."), ("commit", "-qm", "init"), ("push", "-q", "origin", "HEAD:main")):
            git(seed, *cmd)
        self.a = self.clone("a")
        self.b = self.clone("b")

    def clone(self, name):
        path = self.tmp / name
        git(self.tmp, "clone", "-q", str(self.remote), str(path))
        git(path, "config", "user.email", f"{name}@t")
        git(path, "config", "user.name", name)
        return path

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_racing_claims_only_one_wins(self):
        repo_a, repo_b = rs.GitRepo(self.a), rs.GitRepo(self.b)
        task = "exp1-lt--ETTh1--DLinear"
        interleaved = []

        def claim(root, worker):
            def mutate():
                if not interleaved and worker == "b":   # a pushes between b's sync and push
                    interleaved.append(1)
                    repo_a.transaction(claim(self.a, "a"), "a claims")
                if task in rs.read_claims(root):
                    return False
                rs.write_claim(root, task, {"worker": worker, "heartbeat": rs.utcnow()})
                return True
            return mutate

        self.assertFalse(repo_b.transaction(claim(self.b, "b"), "b claims", attempts=3))
        self.assertEqual(rs.read_claims(self.b)[task]["worker"], "a")

    def test_reset_refuses_to_drop_unpushed_code_commits(self):
        (self.a / "code.py").write_text("x = 2\n")
        git(self.a, "commit", "-qam", "local code change")
        with self.assertRaises(rs.UnsafeSync):
            rs.GitRepo(self.a).sync()


if __name__ == "__main__":
    unittest.main()
