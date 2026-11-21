"""Synthetic unit fixtures (not project OOF/test results)."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fusion import sprint5_crop_adapter as a


def detection(score=.5, form="bulk", material="metal"):
    return dict(box=(0, 0, 20, 20), score=score, form=form, material=material,
                volume=10., mass=12., take=1)


WOOD = [.02, .90, .02, .02, .02, .02]
FLAT = [1 / 6] * 6


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "prob.csv"
        self.fold = {"img": 1, "empty": 2}
        self.base = {"img": [detection(.52), detection(.53)], "empty": []}

    def probs(self, fold, change=None):
        fields = a.id_fields(fold)
        header = [*fields, "p_bulk", *(f"p_material_{m}" for m in a.MATERIALS)]
        inv = a.inventory(self.base, fold)
        records = [[*r[:len(fields)], .95, *WOOD] for r in inv]
        if change is not None:
            change(header, records)
        with self.path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(records)
        return a.read_probs(self.path, inv, fold)

    def test_ordered_exact_join_without_empty_image_rows(self):
        self.assertEqual(a.id_fields(self.fold)[:3], ("image_id", "hyp_idx", "fold"))
        for fold in (self.fold, None):
            self.assertEqual(self.probs(fold), {("img", 0): (.95, WOOD), ("img", 1): (.95, WOOD)})

    def test_probability_rejections(self):
        changes = (
            ("unordered", lambda h, r: r.reverse()),
            ("duplicate", lambda h, r: r.append(r[0])),
            ("coverage", lambda h, r: r.pop()),
            ("wrong fold", lambda h, r: r[0].__setitem__(2, 2)),
            ("stale detection", lambda h, r: r[0].__setitem__(3, "0" * 64)),
            ("unknown slot", lambda h, r: r[0].__setitem__(1, "00")),
            ("unknown slot", lambda h, r: r[0].__setitem__(0, "empty")),
            ("nonfinite", lambda h, r: r[0].__setitem__(4, "NaN")),
            ("nonfinite", lambda h, r: r[0].__setitem__(5, "Inf")),
            ("out-of-range", lambda h, r: r[0].__setitem__(5, -1)),
            ("sum to one", lambda h, r: r[0].__setitem__(6, .1)),
            ("expected header", lambda h, r: h.reverse()),
            ("wrong column count", lambda h, r: r[0].pop()),
        )
        for message, fn in changes:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.probs(self.fold, fn)
        with self.assertRaisesRegex(ValueError, "expected header"):
            self.probs(None, lambda h, r: h.insert(2, "fold"))

    def test_scores_shrink_and_ties(self):
        old = [detection(.6), detection(.55)]
        shrunk = a.form_scores(old, [.0, 1.])
        self.assertGreater(shrunk[0], shrunk[1])
        self.assertLess(shrunk[0], .6)
        self.assertGreater(shrunk[1], .55)
        ties = [detection(.6), detection(.6)]
        self.assertEqual(a.form_scores(ties, [0., 1.]), [.6, .6])
        self.assertEqual(a.form_scores(ties, [1., 0.]), [.6, .6])
        self.assertEqual(a.form_scores(ties, [1., 1.])[0], a.form_scores(ties, [1., 1.])[1])
        self.assertGreater(a.form_scores([detection(.5, "solid")], [0.])[0], .5)
        self.assertEqual(a.form_scores([detection(.5)], [.5]), [.5])

    def test_fixed_material_and_isolation(self):
        self.assertEqual(a.material_label("metal", WOOD), "wood")
        self.assertEqual(a.material_label("metal", FLAT), "metal")
        self.assertEqual(a.material_label("metal", [.2, .8, 0, 0, 0, 0]), "metal")
        for variant, expected in (("form", {"score"}), ("material", {"material"}),
                                  ("both", {"score", "material"})):
            p = a.apply(self.base, self.probs(self.fold), variant)
            self.assertEqual(p["empty"], [])
            for d, q in zip(self.base["img"], p["img"]):
                self.assertEqual({k for k in d if d[k] != q[k]}, expected)

    def test_official_ab_support_and_leave_one_object_out(self):
        gt, objs, base, probs, folds = {}, {}, {}, {}, {}
        for k in range(1, 4):
            i, j = f"pos{k}", f"neg{k}"
            gt[i], objs[i], base[i] = [detection(material="wood")], [f"obj{k}"], [detection(.52)]
            gt[j], objs[j], base[j] = [], [], [detection(.54)]
            probs[i, 0], probs[j, 0] = (1., WOOD), (0., FLAT)
            folds[i] = folds[j] = k
        result = a.evaluate(gt, objs, base, probs, folds)
        self.assertEqual(set(result["baseline"]["pooled"]), {*a.KEYS, "AP"})
        self.assertIsNone(result["baseline"]["pooled"]["AP"]["solid"])
        self.assertEqual(result["form"]["delta"]["M"], 0)
        self.assertEqual(result["material"]["delta"]["D"], 0)
        self.assertTrue(result["both"]["positive"])
        self.assertEqual(len(result["both"]["support"]["material_fixed_objects"]), 3)
        self.assertEqual(len(result["both"]["loo"]["all_deltas"]), 3)
        neutral = a.evaluate(gt, objs, base, {k: (.5, FLAT) for k in probs}, folds)
        self.assertTrue(all(not neutral[v]["positive"] for v in a.VARIANTS))

    def test_export_preserves_raw_mass_volume_take(self):
        raw = {"img": "0 0 20 20 .50 bulk metal 1e1 1.20e1 1", "empty": "none"}
        base = {"img": [detection()], "empty": []}
        for v, expected in (("form", [4]), ("material", [6]), ("both", [4, 6])):
            pred = a.apply(base, {("img", 0): (1, WOOD)}, v)
            final = a.export_cells(raw, base, pred, v)
            self.assertEqual(final["empty"], "none")
            self.assertEqual([k for k, (x, y) in enumerate(zip(raw["img"].split(), final["img"].split()))
                              if x != y], expected)

    def test_mask_donors_reuse_opposite_form_anchor(self):
        anchor = detection()
        alt = dict(anchor, form="solid")
        self.assertEqual(a.alternate_slots([anchor, alt], [anchor], [4]), [4, 4])
        with self.assertRaisesRegex(ValueError, "opposite-form"):
            a.alternate_slots([anchor, dict(alt, box=(50, 50, 60, 60))], [anchor], [4])

    def test_attestation_checks_checkpoint_and_split_masks(self):
        o, p, m, ckpt = [Path(self.tmp.name) / f for f in ("base", "probs", "manifest.json", "weights")]
        for f in (o, p, ckpt):
            f.write_bytes(f.name.encode())
        obj = {"train": {"a"}, "hold": {"b"}, "allowed": {"a"}, "denied_synth": {"a"}}
        limits = {"1": {"real": ["train"], "allowed_synth": ["allowed"], "forbidden_object_ids": ["b"]}}
        inputs = {"fixture": "hash"}
        model = dict(train_image_ids=["train", "allowed"], checkpoint_sha256=a.sha(ckpt),
                     checkpoint_path=ckpt.name, inference_crop_source="predicted_mask")
        payload = dict(version=1, split="oof", dataset=inputs, base_sha256=a.sha(o),
                       probabilities_sha256=a.sha(p), models={"1": model})
        def check():
            m.write_text(json.dumps(payload))
            return a.attest(m, "oof", o, p, obj, inputs, limits)
        result = check()
        self.assertTrue(result["models"]["1"]["checkpoint_bytes_verified"])
        self.assertFalse(result["model_provenance_proven"])
        self.assertEqual(a.attest(None, "oof", o, p, obj, inputs, limits)["status"], "unattested")
        for ids in (["hold"], ["denied_synth"]):
            model["train_image_ids"] = ids
            with self.assertRaisesRegex(ValueError, "unapproved training"):
                check()
        model["train_image_ids"] = ["train"]
        model["inference_crop_source"] = "ground_truth"
        with self.assertRaisesRegex(ValueError, "invalid crop"):
            check()
        model["inference_crop_source"] = "predicted_box"
        p.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "input hashes"):
            check()

    def test_export_refuses_nonpositive_oof(self):
        output = Path(self.tmp.name) / "not-created.csv"
        with patch.object(a, "training_contract", return_value=(None,) * 5), \
             patch.object(a, "scored", return_value={"variants": {"both": {"positive": False, "gate": {}}}}), \
             patch.object(a, "load_test") as load_test, patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                a.cli(["export", "--variant", "both", "--deploy-spec", "unused", "--oof-probs", "unused",
                       "--test-probs", "unused", "--out", str(output)])
        load_test.assert_not_called()
        self.assertFalse(output.exists())

    def test_positive_export_requires_explicit_provenance_override(self):
        raw = {"img": "0 0 20 20 .50 bulk metal 1e1 1.20e1 1"}
        base = {"img": [detection()]}
        probs = {("img", 0): (1., WOOD)}
        output = Path(self.tmp.name) / "candidate.csv"
        report = {"variants": {"form": {"positive": True, "gate": {"fixture": True}, "delta": {"Score": .01}}},
                  "attestation": {"status": "unattested"}}
        argv = ["export", "--variant", "form", "--deploy-spec", "unused", "--oof-probs", "unused",
                "--test-probs", "unused", "--out", str(output)]
        with patch.object(a, "training_contract", return_value=(None,) * 5), \
             patch.object(a, "scored", return_value=report), \
             patch.object(a, "verify_deploy_spec", return_value="fixture"), \
             patch.object(a, "load_test", return_value=(raw, base)), \
             patch.object(a, "read_probs", return_value=probs), \
             patch("sys.stderr"), patch("sys.stdout"):
            with self.assertRaises(SystemExit):
                a.cli(argv)
            self.assertFalse(output.exists())
            a.cli([*argv, "--allow-unattested"])
        with output.open() as f:
            row, = csv.DictReader(f)
        self.assertEqual([i for i, (x, y) in enumerate(zip(raw["img"].split(), row["prediction"].split()))
                          if x != y], [4])

    def test_deploy_spec_rejects_hash_drift(self):
        self.path.write_bytes(b"fixture")
        spec = self.path.with_name("spec")
        args = SimpleNamespace(deploy_spec=spec, variant="form", oof_base=self.path,
                               oof_probs=self.path, test_probs=self.path,
                               oof_attestation=None, test_attestation=None)
        h = a.sha(self.path)
        body = dict(version=1, variant="form", config=a.CONFIG, oof_base_sha256=h,
                    test_base_sha256=a.PINNED[a.TEST_BASE], oof_probabilities_sha256=h,
                    test_probabilities_sha256=h, oof_attestation_sha256=None, test_attestation_sha256=None)
        spec.write_text(json.dumps(body))
        with patch.object(a, "TEST_BASE", self.path), patch.dict(a.PINNED, {self.path: h}):
            body["test_base_sha256"] = h
            spec.write_text(json.dumps(body))
            self.assertEqual(a.verify_deploy_spec(args), a.sha(spec))
            args.variant = "material"
            with self.assertRaisesRegex(ValueError, "Deploy spec mismatch"):
                a.verify_deploy_spec(args)
            plan = spec.with_name("plan.json")
            argv = ["plan", "--variant", "form", "--oof-base", str(self.path),
                    "--oof-probs", str(self.path), "--test-probs", str(self.path), "--out", str(plan)]
            a.cli(argv)
            self.assertEqual(a.read_json(plan), body)
            with patch("sys.stderr"), self.assertRaises(SystemExit):
                a.cli(argv)  # plan does not overwrite


@unittest.skipUnless(os.environ.get("SPRINT5_ORACLE_SMOKE") == "1",
                     "opt-in GT oracle smoke; not a model or an OOF estimate")
class FrozenOracleSmoke(unittest.TestCase):
    def test_oracle_material_cli_end_to_end(self):
        """GT is used ONLY to construct this throwaway positive-control fixture."""
        fold, *_ = a.training_contract()
        gt, oof = a.load_oof(a.OOF_BASE, fold)
        frozen_raw, test = a.load_test()
        original_hash = a.sha(a.TEST_BASE)
        oof_labels = {}
        for image, ds in oof.items():
            oof_labels.update({(image, j): d["material"] for j, d in enumerate(ds)})
            for g, j in zip(gt[image], a.slots(gt[image], ds), strict=True):
                if j is not None:
                    oof_labels[image, j] = g["material"]
        test_labels = {(image, j): d["material"] for image, ds in test.items() for j, d in enumerate(ds)}
        forced = set(list(test_labels)[:2])
        for key in forced:
            test_labels[key] = "wood" if test_labels[key] != "wood" else "metal"

        with TemporaryDirectory(prefix="sprint5-oracle-only-") as tmp:
            tmp = Path(tmp)
            for split, base, folds, labels in (("oof", oof, fold, oof_labels),
                                                ("test", test, None, test_labels)):
                keys = a.id_fields(folds)
                header = (*keys, "p_bulk", *(f"p_material_{m}" for m in a.MATERIALS))
                values = [(*r[:len(keys)], .5, *(.99 if m == labels[r[:2]] else .002 for m in a.MATERIALS))
                          for r in a.inventory(base, folds)]
                a.write_csv(tmp / f"ORACLE_FIXTURE_{split}.csv", header, values)

            def timed(*argv):
                start = perf_counter()
                process = subprocess.run([sys.executable, "-B", *map(str, argv)], cwd=a.ROOT,
                                         capture_output=True, text=True)
                elapsed = perf_counter() - start
                # Never propagate stdout from export: it contains the oracle-only delta.
                self.assertEqual(process.returncode, 0, process.stderr)
                return process.stdout, elapsed

            script = Path(a.__file__)
            spec = tmp / "ORACLE_FIXTURE_plan.json"
            out = tmp / "ORACLE_FIXTURE_DO_NOT_SUBMIT.csv"
            common = ("--variant", "material", "--oof-probs", tmp / "ORACLE_FIXTURE_oof.csv",
                      "--test-probs", tmp / "ORACLE_FIXTURE_test.csv")
            _, plan_s = timed(script, "plan", *common, "--out", spec)
            export, export_s = timed(script, "export", *common, "--deploy-spec", spec,
                                     "--allow-unattested", "--out", out)
            receipt = json.loads(export)
            self.assertEqual(receipt["output_sha256"], a.sha(out))
            self.assertEqual(receipt["oof_attestation"], "unattested")
            self.assertEqual(receipt["test_attestation"], "unattested")
            validation, check_s = timed(a.WORK / "submission_tools.py", out)
            self.assertIn("137 frames, 342 objects, 10 none", validation)
            result = dict(a.rows(out, ("image_id", "prediction")))
            self.assertEqual(list(result), list(frozen_raw))
            edits = set()
            for image in result:
                for j, (before, after) in enumerate(zip(frozen_raw[image].split(";"),
                                                        result[image].split(";"), strict=True)):
                    for k, (old, new) in enumerate(zip(before.split(), after.split(), strict=True)):
                        if old != new:
                            edits.add((image, j, k))
            self.assertEqual(edits, {(image, j, 6) for image, j in forced})
            self.assertEqual(a.sha(a.TEST_BASE), original_hash)
            print(f"\nORACLE TEST ONLY: plan {plan_s:.3f}s; export {export_s:.3f}s; "
                  f"validator {check_s:.3f}s; 137/342/10, exactly two material tokens changed.")


if __name__ == "__main__":
    unittest.main()
