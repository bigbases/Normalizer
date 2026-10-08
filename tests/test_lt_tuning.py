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
        final = c["stage"] in ("final", "explore")
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


class LightNormExploreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = rm.load_json(rm.PROTOCOL_PATH)
        cls.bases = rm.load_json(rm.BASE_CONFIGS_PATH)

    def test_explore_cells_override_supplied_and_evaluate_test(self):
        base = rm.selected_base_config(self.bases, "ETTm1", "DLinear")
        doc = {"cases": {"ETTm1|DLinear": [
            {"id": "a", "params": {"kernel_size": 49}},
            {"id": "b", "params": {"station_lr": 0.0005}, "horizons": [96, 192, 336, 720],
             "seeds": [2021, 2022, 2023]}]}}
        schedule = rm.explore_candidates(self.protocol, "ETTm1", "DLinear", base, doc)
        self.assertEqual(schedule[0][0], {**rm.supplied_lt_params(base), "kernel_size": 49})
        self.assertEqual((schedule[0][1], schedule[0][2]), ([96, 720], [2021]))
        self.assertEqual(len(schedule[1][1]) * len(schedule[1][2]), 12)
        cells = rs._cells(self.protocol, self.bases, "6_lightnorm_explore", "explore", "ETTm1", "DLinear", ["lt"])
        self.assertTrue(cells and all(c["run_id"].startswith("explore-") for c in cells))
        cmd = rm.cell_command(cells[0], "/data", "/tmp/r.csv", "/tmp/ck")
        self.assertIn("explore", cmd)
        self.assertNotIn("--skip_test", cmd)

    def test_explore_summary_compares_on_same_seeds(self):
        rows = {}
        for dataset, backbone in (("ETTm1", "DLinear"),):
            rows.update(fake_rows(rs._cells(self.protocol, self.bases, "6_lightnorm_explore", "explore",
                                            dataset, backbone, ["lt"])))
            for method in ("lt", "san", "ddn"):
                exp = "1_rebuttal_completion" if method == "lt" else "3_frozen_backbone_comparison"
                doc = {"locks": {f"{dataset}|{backbone}|{m}": {} for m in ("san", "ddn")}}
                rows.update(fake_rows(rs._cells(self.protocol, self.bases, exp, "final", dataset, backbone,
                                                [method], lock_doc=doc)))
        out = Path(tempfile.mkdtemp())
        st.exp6_lt_explore(out, list(rows.values()), self.protocol, self.bases)
        with (out / "exp6_lt_explore.csv").open() as f:
            table = [r for r in csv.DictReader(f) if r["Dataset"] == "ETTm1" and r["Backbone"] == "DLinear"]
        self.assertTrue(table)
        self.assertTrue(all(len(set(r["Done"].split("/"))) == 1 and r["SAN_MSE"] and r["vs_DDN_pct"]
                            for r in table))


class LightNormMainTuningTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = rm.load_json(rm.PROTOCOL_PATH)
        cls.bases = rm.load_json(rm.BASE_CONFIGS_PATH)
        cls.plan = rs.load_plan()

    def cells(self, experiment, stage, dataset, backbone, lock_doc=None):
        return rs._cells(self.protocol, self.bases, experiment, stage, dataset, backbone, ["lt"], lock_doc=lock_doc)

    def test_grid_contains_supplied_and_search_skips_it(self):
        for dataset, backbone in (("ETTm1", "DLinear"), ("ETTh2", "iTransformer"), ("Traffic", "iTransformer")):
            base = rm.selected_base_config(self.bases, dataset, backbone)
            grid = rm.lt_main_candidates(self.protocol, base)
            self.assertEqual(len(grid), 6)
            self.assertEqual(sum(rm.is_supplied_lt(p, base) for p in grid), 1)
            self.assertEqual({p["station_lr"] for p in grid}, {1e-4, 5e-4, 1e-3})
            self.assertIn(49, {p["kernel_size"] for p in grid})
            search = self.cells("7_lightnorm_tuning", "search", dataset, backbone)
            self.assertEqual(len(search), 10)

    def test_lock_uses_supplied_final_validation_and_reuses_exp1_cells(self):
        rows = fake_rows(self.cells("7_lightnorm_tuning", "search", "ETTm1", "DLinear"), value=0.6)
        exp1 = self.cells("1_rebuttal_completion", "final", "ETTm1", "DLinear")
        rows.update(fake_rows(exp1, value=0.1, step=0))         # supplied is best on validation
        doc = sh.lt_locks(rows.values(), self.protocol, self.bases, [("ETTm1", "DLinear")])
        base = rm.selected_base_config(self.bases, "ETTm1", "DLinear")
        self.assertTrue(rm.is_supplied_lt(doc["locks"]["ETTm1|DLinear|lt"], base))
        tuned = self.cells("8_lightnorm_tuned_final", "final", "ETTm1", "DLinear", lock_doc=doc)
        self.assertEqual({c["run_id"] for c in tuned}, {c["run_id"] for c in exp1})

        tasks = {t.task_id: t for t in rs.build_tasks(self.plan, self.protocol, self.bases, rows)}
        self.assertEqual(tasks["lt-search--ETTm1--DLinear"].status, "done")
        self.assertEqual(tasks["lt-tuned--ETTm1--DLinear"].status, "done")
        self.assertEqual(tasks["lt-tuned--Weather--DLinear"].status, "blocked")
        self.assertLess(tasks["lt-search--ETTm1--iTransformer"].rank, tasks["lt-search--ETTm2--DLinear"].rank)

    def test_non_supplied_lock_is_reported_as_lt_tuned(self):
        rows = fake_rows(self.cells("7_lightnorm_tuning", "search", "ETTm1", "DLinear"), value=0.1)
        rows.update(fake_rows(self.cells("1_rebuttal_completion", "final", "ETTm1", "DLinear"), value=0.9))
        doc = sh.lt_locks(rows.values(), self.protocol, self.bases, [("ETTm1", "DLinear")])
        tuned = fake_rows(self.cells("8_lightnorm_tuned_final", "final", "ETTm1", "DLinear", lock_doc=doc))
        label = st.method_labeler(self.bases)
        self.assertEqual({label(r) for r in tuned.values()}, {"lt_tuned"})
        rows.update(tuned)
        groups = st.final_groups(list(rows.values()), label)
        self.assertEqual(len(groups[("ETTm1", "DLinear", "lt_tuned", 96)]), 3)
        self.assertEqual(len(groups[("ETTm1", "DLinear", "lt", 96)]), 3)


if __name__ == "__main__":
    unittest.main()
