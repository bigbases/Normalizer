import csv
import tempfile
import unittest
from pathlib import Path

from experiments import result_store as rs
from experiments import run_matrix as rm
from experiments import select_hparams as sh
from experiments import summary_tables as st


def fake_rows(cells, value=0.5, step=1e-3):
    rows = {}
    for i, c in enumerate(cells):
        final = c["stage"] == "final"
        rows[c["run_id"]] = dict(
            RunID=c["run_id"], CandidateID=c["candidate_id"], ConfigHash=c["config_hash"],
            Phase=c["stage"], Split="test" if final else "validation",
            Dataset=c["dataset"], Backbone=c["backbone"], Horizon=str(c["horizon"]),
            Seed=str(c["seed"]), BestValMSE=str(value + i * step), UseNorm=c["method"],
            MSE="0.3" if final else "", MAE="0.4" if final else "",
        )
    return rows


class LightNormPilotTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = rm.load_json(rm.PROTOCOL_PATH)
        cls.bases = rm.load_json(rm.BASE_CONFIGS_PATH)
        cls.plan = rs.load_plan()

    def cells(self, experiment, stage, dataset, backbone, methods=("lt",)):
        return rs._cells(self.protocol, self.bases, experiment, stage, dataset, backbone, list(methods))

    def test_station_lr_ladder_neighbours(self):
        ladder = self.protocol["lt_tuning"]["station_lr_ladder"]
        self.assertEqual(rm.station_lr_neighbors(1e-4, ladder), (5e-05, 1e-4, 5e-4))
        self.assertEqual(rm.station_lr_neighbors(5e-4, ladder), (1e-4, 5e-4, 1e-3))

    def test_pilot_grid_is_factorial_plus_centres(self):
        for dataset, backbone in self.protocol["lt_tuning"]["pilot"]["cases"]:
            base = rm.selected_base_config(self.bases, dataset, backbone)
            grid = rm.lt_pilot_candidates(self.protocol, dataset, base)
            self.assertEqual(len(grid), 10)
            self.assertEqual(len({rm.canonical_hash(p) for p in grid}), 10)
            self.assertEqual({p["use_mlp"] for p in grid}, {int(dataset == "Weather")})
            self.assertEqual({p["down_ratio"] for p in grid}, {base["down_ratio"]})
            self.assertEqual({p["kernel_size"] for p in grid}, {13, 25, 49})
            self.assertEqual({p["station_lr"] for p in grid}, {5e-05, 1e-4, 5e-4})

    def test_supplied_setting_is_a_pilot_centre_when_rule_keeps_use_mlp(self):
        final = {c["candidate_id"] for c in self.cells("1_rebuttal_completion", "final", "ETTm1", "iTransformer")}
        pilot = {c["candidate_id"] for c in self.cells("5_lightnorm_tuning", "search", "ETTm1", "iTransformer")}
        self.assertEqual(len(final), 1)
        self.assertTrue(final <= pilot)
        # use_mlp rule changes ETTm1 x DLinear (supplied use_mlp = 1).
        final = {c["candidate_id"] for c in self.cells("1_rebuttal_completion", "final", "ETTm1", "DLinear")}
        pilot = {c["candidate_id"] for c in self.cells("5_lightnorm_tuning", "search", "ETTm1", "DLinear")}
        self.assertFalse(final & pilot)

    def test_pilot_tasks_and_baseline_selection_ignore_lt(self):
        tasks = {t.task_id: t for t in rs.build_tasks(self.plan, self.protocol, self.bases, {})}
        pilot = [t for t in tasks.values() if t.phase == "lt-pilot"]
        self.assertEqual(sorted(t.task_id for t in pilot), [
            "lt-pilot--ETTm1--DLinear", "lt-pilot--ETTm1--iTransformer", "lt-pilot--Weather--DLinear"])
        self.assertTrue(all(t.status == "ready" and len(t.cells) == 20 for t in pilot))
        self.assertTrue(all(c["stage"] == "search" and c["seed"] == 2021 for t in pilot for c in t.cells))

        rows = fake_rows(self.cells("5_lightnorm_tuning", "search", "ETTm1", "DLinear"))
        rows.update(fake_rows(self.cells("2_normalizer_search", "search", "ETTm1", "DLinear", ["san"])))
        doc = sh.select(sh.validation_rows(rows.values()), "shortlist", self.protocol, self.bases)
        self.assertEqual(list(doc["shortlists"]), ["ETTm1|DLinear|san"])

    def test_pilot_summary_effects(self):
        rows = {}
        for dataset, backbone in self.protocol["lt_tuning"]["pilot"]["cases"]:
            rows.update(fake_rows(self.cells("5_lightnorm_tuning", "search", dataset, backbone)))
            rows.update(fake_rows(self.cells("1_rebuttal_completion", "final", dataset, backbone)))
        out = Path(tempfile.mkdtemp())
        st.exp5_lt_pilot(out, list(rows.values()), self.protocol, self.bases)
        with (out / "exp5_lt_pilot.csv").open() as f:
            table = list(csv.DictReader(f))
        with (out / "exp5_lt_pilot_effects.csv").open() as f:
            effects = list(csv.DictReader(f))
        self.assertEqual(len(table), 3 * 11)
        self.assertEqual(sum(r["Point"] == "supplied" for r in table), 3)
        self.assertTrue(all(r["Val_mean"] for r in table))
        # 3 cases x 2 horizons x (3 main + 3 two-way + centre) terms
        self.assertEqual(len(effects), 3 * 2 * 7)
        self.assertTrue(all(r["Noise_pct"] for r in effects))


if __name__ == "__main__":
    unittest.main()
