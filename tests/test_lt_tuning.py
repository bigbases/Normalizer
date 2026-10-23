import csv
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from experiments import fill_missing as fm
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
        # The pilot (dev, Exp5) keeps the dev-era supplied setting; Exp1 ties kernel_len to kernel_size.
        base = rm.selected_base_config(self.bases, "ETTm1", "iTransformer")
        dev_supplied = rm.canonical_hash({"method": "lt", "params": rm.complete_method_params("lt", rm.supplied_lt_params(base))})
        pilot = {c["candidate_id"] for c in self.cells("5_lightnorm_tuning", "search", "ETTm1", "iTransformer")}
        self.assertIn(dev_supplied, pilot)
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

    def test_selection_ignores_validation_cells_of_retired_backbones(self):
        # TimeMixer++ search cells stay in the store after the backbone was retired.
        rows = fake_rows(self.cells("2_normalizer_search", "search", "ETTm1", "DLinear", ["san"]))
        for i, cid in enumerate(("e082b1250cbb", "f00000000000")):
            for h in (96, 720):
                rid = f"search-ETTh1-TimeMixerPP-ddn-h{h}-s2021-c{cid}"
                rows[rid] = dict(RunID=rid, CandidateID=cid, Phase="search", Split="validation", Dataset="ETTh1",
                                 Backbone="TimeMixerPP", Horizon=str(h), Seed="2021", BestValMSE=str(0.4 + i),
                                 UseNorm="ddn", MSE="", MAE="")
        for mode, key in (("shortlist", "shortlists"), ("lock", "locks")):
            doc = sh.select(sh.validation_rows(rows.values()), mode, self.protocol, self.bases)
            self.assertFalse([k for k in doc[key] if "TimeMixerPP" in k])

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

    def test_grid_and_search_skip_the_supplied_point(self):
        # ETTh2 x iTransformer was supplied with station_lr 5e-4, which is not a grid level
        for dataset, backbone, supplied in (("ETTm1", "DLinear", 1), ("ETTh2", "iTransformer", 0),
                                            ("Traffic", "iTransformer", 1)):
            base = rm.selected_base_config(self.bases, dataset, backbone)
            grid = rm.lt_main_candidates(self.protocol, base)
            self.assertEqual({(p["kernel_size"], p["station_lr"]) for p in grid},
                             {(k, lr) for k in (13, 25, 49) for lr in (1e-4, 1e-3)})
            self.assertEqual(sum(rm.is_supplied_lt(p, base) for p in grid), supplied)
            search = self.cells("7_lightnorm_tuning", "search", dataset, backbone)
            self.assertEqual(len(search), 2 * (6 - supplied))

    def confirm_rows(self, rows, dataset, backbone, value):
        doc = sh.lt_shortlists(rows.values(), self.protocol, self.bases, [(dataset, backbone)])
        cells = rs._cells(self.protocol, self.bases, "7_lightnorm_tuning", "confirm", dataset, backbone, ["lt"],
                          shortlist_doc=doc)
        return fake_rows(cells, value=value, step=0)

    def test_kernel_len_follows_kernel_size_and_keeps_aligned_ids(self):
        for c in self.cells("7_lightnorm_tuning", "search", "ETTm1", "iTransformer"):
            self.assertEqual(c["effective"]["kernel_len"], c["effective"]["kernel_size"])
        # ETTm1 x DLinear already had kernel_len == kernel_size, so its Exp1 cells keep their IDs.
        base = rm.selected_base_config(self.bases, "ETTm1", "DLinear")
        old = rm.canonical_hash({"method": "lt", "params": rm.complete_method_params("lt", rm.supplied_lt_params(base))})
        self.assertEqual({c["candidate_id"] for c in self.cells("1_rebuttal_completion", "final", "ETTm1", "DLinear")}, {old})

    def test_lock_uses_supplied_final_validation_and_reuses_exp1_cells(self):
        rows = fake_rows(self.cells("7_lightnorm_tuning", "search", "ETTm1", "DLinear"), value=0.6)
        exp1 = self.cells("1_rebuttal_completion", "final", "ETTm1", "DLinear")
        rows.update(fake_rows(exp1, value=0.1, step=0))         # supplied is best on validation
        self.assertEqual(sh.lt_locks(rows.values(), self.protocol, self.bases, [("ETTm1", "DLinear")])["locks"], {})
        confirm = self.confirm_rows(rows, "ETTm1", "DLinear", 0.6)
        # the supplied point takes seed 2022 from Exp1; only the other shortlisted point is confirmed
        self.assertEqual(len(confirm), 2)
        self.assertEqual({r["Seed"] for r in confirm.values()}, {"2022"})
        rows.update(confirm)
        doc = sh.lt_locks(rows.values(), self.protocol, self.bases, [("ETTm1", "DLinear")])
        base = rm.selected_base_config(self.bases, "ETTm1", "DLinear")
        self.assertTrue(rm.is_supplied_lt(doc["locks"]["ETTm1|DLinear|lt"], base))
        tuned = self.cells("8_lightnorm_tuned_final", "final", "ETTm1", "DLinear", lock_doc=doc)
        self.assertEqual({c["run_id"] for c in tuned}, {c["run_id"] for c in exp1})

        tasks = {t.task_id: t for t in rs.build_tasks(self.plan, self.protocol, self.bases, rows)}
        self.assertEqual(tasks["lt-search--ETTm1--DLinear"].status, "done")
        self.assertEqual(tasks["lt-confirm--ETTm1--DLinear"].status, "done")
        self.assertEqual(tasks["lt-tuned--ETTm1--DLinear"].status, "done")
        self.assertEqual(tasks["lt-tuned--Weather--DLinear"].status, "blocked")
        self.assertLess(tasks["lt-search--ETTm1--iTransformer"].rank, tasks["lt-search--ETTm2--DLinear"].rank)

    def test_kernel_sensitivity_varies_only_the_kernel_of_the_lock(self):
        base = rm.selected_base_config(self.bases, "ETTh1", "DLinear")
        lock = {**rm.supplied_lt_params(base), "kernel_size": 49, "station_lr": 1e-3}
        doc = {"locks": {"ETTh1|DLinear|lt": lock}}
        sweep = self.cells("12_lightnorm_kernel_sensitivity", "final", "ETTh1", "DLinear", lock_doc=doc)
        kernels = self.protocol["experiments"]["12_lightnorm_kernel_sensitivity"]["kernel_sweep"]
        self.assertEqual(len(sweep), len(kernels) * len(self.protocol["seeds"]))   # H=96 only
        self.assertEqual({c["horizon"] for c in sweep}, {96})
        self.assertEqual({c["effective"]["kernel_size"] for c in sweep}, set(kernels))
        for c in sweep:
            self.assertEqual(c["effective"]["kernel_len"], c["effective"]["kernel_size"])
            others = {k: v for k, v in c["effective"].items() if k not in ("kernel_size", "kernel_len")}
            self.assertEqual(others, {k: v for k, v in rm.lt_params(lock).items() if k not in ("kernel_size", "kernel_len")})
        # the locked kernel reuses the tuned final cells of experiment 8
        tuned = {c["run_id"] for c in self.cells("8_lightnorm_tuned_final", "final", "ETTh1", "DLinear", lock_doc=doc)}
        self.assertEqual({c["run_id"] for c in sweep if c["effective"]["kernel_size"] == 49}, {r for r in tuned if "-h96-" in r})

    def test_non_supplied_lock_is_reported_as_lt_tuned(self):
        rows = fake_rows(self.cells("7_lightnorm_tuning", "search", "ETTm1", "DLinear"), value=0.1)
        rows.update(fake_rows(self.cells("1_rebuttal_completion", "final", "ETTm1", "DLinear"), value=0.9))
        confirm = self.confirm_rows(rows, "ETTm1", "DLinear", 0.1)
        self.assertEqual(len(confirm), 4)                       # two new points x horizons 96/720
        rows.update(confirm)
        doc = sh.lt_locks(rows.values(), self.protocol, self.bases, [("ETTm1", "DLinear")])
        tuned = fake_rows(self.cells("8_lightnorm_tuned_final", "final", "ETTm1", "DLinear", lock_doc=doc))
        label = st.method_labeler(self.bases)
        self.assertEqual({label(r) for r in tuned.values()}, {"lt_tuned"})
        rows.update(tuned)
        groups = st.final_groups(list(rows.values()), label)
        self.assertEqual(len(groups[("ETTm1", "DLinear", "lt_tuned", 96)]), 3)
        self.assertEqual(len(groups[("ETTm1", "DLinear", "lt", 96)]), 3)

    def test_t3_structure_search_follows_the_best_grid_point(self):
        grid = self.cells("7_lightnorm_tuning", "search", "ETTh1", "TimeXer")
        self.assertEqual(len(grid), 12)                         # 6 points x 2 horizons, supplied included
        rows = fake_rows(grid, value=0.5)
        anchor = sh.lt_grid_best(rows.values(), self.protocol, self.bases, [("ETTh1", "TimeXer")])
        best = anchor["locks"]["ETTh1|TimeXer|lt"]
        self.assertEqual(rm.lt_params(best), rm.lt_params(grid[0]["effective"]))   # lowest fake value
        struct = self.cells("9_lightnorm_structure", "search", "ETTh1", "TimeXer", lock_doc=anchor)
        self.assertEqual(len(struct), 6)                        # 3 s_norm x use_mlp variants x 2 horizons
        self.assertEqual({(c["effective"]["s_norm"], c["effective"]["use_mlp"]) for c in struct}, {(1, 0), (0, 1), (1, 1)})
        self.assertTrue(all(c["effective"]["kernel_size"] == best["kernel_size"] for c in struct))
        self.assertEqual(sh.lt_shortlists(rows.values(), self.protocol, self.bases, [("ETTh1", "TimeXer")])["shortlists"], {})
        rows.update(fake_rows(struct, value=0.1))               # a structure variant wins
        confirm = self.confirm_rows(rows, "ETTh1", "TimeXer", 0.1)
        self.assertEqual(len(confirm), 4)
        rows.update(confirm)
        lock = sh.lt_locks(rows.values(), self.protocol, self.bases, [("ETTh1", "TimeXer")])["locks"]["ETTh1|TimeXer|lt"]
        self.assertNotEqual((lock["s_norm"], lock["use_mlp"]), (0, 0))
        tasks = {t.task_id: t for t in rs.build_tasks(self.plan, self.protocol, self.bases, rows)}
        for phase in ("t3-lt-search", "t3-lt-struct", "t3-lt-confirm"):
            self.assertEqual(tasks[f"{phase}--ETTh1--TimeXer"].status, "done")
        self.assertEqual(tasks["t3-lt-tuned--ETTh1--TimeXer"].status, "ready")
        self.assertEqual(tasks["t3-lt-struct--ETTm1--TimeXer"].status, "blocked")


class LightNormKernelCheckTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = rm.load_json(rm.PROTOCOL_PATH)
        cls.bases = rm.load_json(rm.BASE_CONFIGS_PATH)
        cls.plan = rs.load_plan()

    def test_check_cells_are_points_of_the_13_25_49_grid(self):
        cells = rs._cells(self.protocol, self.bases, "10_lightnorm_kernel_check", "search", "Traffic", "DLinear", ["lt"])
        self.assertEqual(len(cells), 4)                      # 2 station_lr x horizons 96/720 at seed 2021
        self.assertEqual({(c["effective"]["kernel_size"], c["effective"]["kernel_len"]) for c in cells}, {(13, 13)})
        self.assertEqual({c["effective"]["station_lr"] for c in cells}, {1e-4, 1e-3})
        grid = rs._cells(self.protocol, self.bases, "7_lightnorm_tuning", "search", "Traffic", "DLinear", ["lt"])
        self.assertLessEqual({c["run_id"] for c in cells}, {c["run_id"] for c in grid})

    def test_open_phase_is_claimed_whatever_the_phase_filter(self):
        tasks = [t for t in rs.build_tasks(self.plan, self.protocol, self.bases, {}) if t.phase == "lt-kernel-check"]
        self.assertEqual(len(tasks), 4)
        big = Namespace(phases="t3-search", datasets=None, backbones=None, open_phases=fm.open_phases(self.plan))
        small = Namespace(phases="t3-search", datasets="ETTh1,Weather", backbones=None, open_phases=big.open_phases)
        self.assertTrue(all(fm.task_matches(t, big) for t in tasks))
        self.assertFalse(any(fm.task_matches(t, small) for t in tasks))   # dataset filters still apply


class SanGridTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = rm.load_json(rm.PROTOCOL_PATH)
        cls.bases = rm.load_json(rm.BASE_CONFIGS_PATH)

    def cells(self, stage, shortlist_doc=None):
        return rs._cells(self.protocol, self.bases, "2_normalizer_search", stage, "ETTm1", "DLinear", ["san"],
                         shortlist_doc=shortlist_doc)

    def test_grid_uses_official_period_lens_and_keeps_old_ids(self):
        base = rm.selected_base_config(self.bases, "ETTm1", "DLinear")
        grid = rm.search_candidates(self.protocol, "ETTm1", "san", base)
        self.assertEqual({(p["period_len"], p["station_lr"]) for p in grid},
                         {(p, lr) for p in (12, 24) for lr in (5e-5, 1e-4, 5e-4)})
        # period_len 24 points of the old station_lr-scale grid keep their run identity
        old = {rm.canonical_hash({"method": "san", "params": rm.complete_method_params(
            "san", rm.normalized_params("san", {"period_len": 24, "station_lr_scale": s}, base))}) for s in (0.5, 1.0)}
        self.assertLessEqual(old, {c["candidate_id"] for c in self.cells("search")})

    def test_lock_is_chosen_among_the_shortlist_only(self):
        search = self.cells("search")
        rows = fake_rows(search)
        shortlist = sh.select(sh.validation_rows(rows.values()), "shortlist", self.protocol, self.bases)
        top2 = {c["candidate_id"] for c in shortlist["shortlists"]["ETTm1|DLinear|san"]}
        confirm = fake_rows(self.cells("confirm", shortlist_doc=shortlist), value=0.5, step=0)
        rows.update(confirm)
        # confirm cells left by a candidate shortlisted under an earlier grid look better
        outsider = next(c["candidate_id"] for c in search if c["candidate_id"] not in top2)
        for rid, row in list(confirm.items()):
            rows["stale-" + rid] = dict(row, RunID="stale-" + rid, CandidateID=outsider, BestValMSE="0.01")
        lock = sh.select(sh.validation_rows(rows.values()), "lock", self.protocol, self.bases)["locks"]["ETTm1|DLinear|san"]
        self.assertIn(rm.canonical_hash({"method": "san", "params": lock}), top2)


if __name__ == "__main__":
    unittest.main()
