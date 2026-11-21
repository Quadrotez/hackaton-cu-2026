"""Replay pinned Gregory DINO both and stack it with the saved take mixture."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from fusion.instance_take_ab import apply_mixture
from fusion import sprint5_crop_adapter as adapter


def git_file(revision, filename):
    return subprocess.check_output(["git", "show", f"{revision}:{filename}"], cwd=adapter.ROOT)


def sha_bytes(contents):
    return hashlib.sha256(contents).hexdigest()


def overlay_take(dino, mixture):
    """Merge aligned slots explicitly; the take path must use untouched input."""
    assert list(dino) == list(mixture)
    out = {}
    for i in dino:
        out[i] = []
        for d, t in zip(dino[i], mixture[i], strict=True):
            assert d.keys() == t.keys()
            assert all(d[k] == t[k] for k in d.keys() - {"material", "score", "take"})
            out[i].append(dict(d, take=t["take"]))
    return out


def replay_mixture(base, fold, path, policy):
    inventory = adapter.inventory(base, fold)
    assert policy["inputs"]["baseline_sha"] == adapter.OOF_CANONICAL
    assert policy["inputs"]["embeddings_sha"] == adapter.sha(path)
    with np.load(path, allow_pickle=False) as z:
        assert z["keys"].tolist() == [[i, str(j), sha] for i, j, _, sha, *_ in inventory]
        ix = z["donor_index"]
        views = [z[k][ix] for k in ("d", "c", "df", "cf")]
    out = {i: [] for i in base}
    for k in sorted(set(fold.values())):
        sel = [n for n, row in enumerate(inventory) if row[2] == k]
        slots = [inventory[n][:2] for n in sel]
        rows = [base[i][j] for i, j in slots]
        for (i, j), d in zip(slots, apply_mixture(rows, [v[sel] for v in views],
                                                 policy["folds"][str(k)]), strict=True):
            assert len(out[i]) == j
            out[i].append(d)
    return out


def verify_components(gt, base, dino, mixture, stack):
    for i in gt:
        old = adapter.slots(gt[i], base[i])
        assert all(adapter.slots(gt[i], v[i]) == old for v in (dino, mixture, stack))
        for b, d, m, s in zip(base[i], dino[i], mixture[i], stack[i], strict=True):
            assert all(b[k] == d[k] for k in b.keys() - {"score", "material"})
            assert all(b[k] == m[k] for k in b.keys() - {"take"})
            assert all(d[k] == s[k] for k in d.keys() - {"take"}) and s["take"] == m["take"]
    b, d, m, s = [adapter.metrics(gt, v) for v in (base, dino, mixture, stack)]
    assert s["D"] == d["D"] and s["M"] == d["M"] and s["S"] == m["S"]
    assert s["V"] == d["V"] == m["V"] == b["V"]
    assert abs(s["Score"] - (d["Score"] + m["Score"] - b["Score"])) < 1e-10
    return dict(base=b, dino_both=d, mixture=m, stack=s, delta_base=adapter.delta(b, s),
                delta_mixture=adapter.delta(m, s), delta_dino=adapter.delta(d, s))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attached-test", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "--verify", "7b5c60e^{commit}"],
                                       cwd=adapter.ROOT, text=True).strip()
    # Exact logic, rather than relying on an interchangeable name like 'blend'.
    assert git_file(revision, "fusion/sprint5_crop_adapter.py") == Path(adapter.__file__).read_bytes()
    upstream = json.loads(git_file(revision, "work/dino_score.json"))
    contract = adapter.training_contract()
    fold, _, _, inputs, _ = contract
    assert upstream["dataset"] == inputs and upstream["config"] == adapter.CONFIG
    assert upstream["oof_base_sha256"] == adapter.sha(adapter.OOF_BASE)
    gt, base = adapter.load_oof(adapter.OOF_BASE, fold)
    root = adapter.WORK / "instance_take_oof"
    mixture_path, policy_path = root / "predictions/mixture.json", root / "mixture_policy.json"
    mixture = replay_mixture(base, fold, root / "predicted_embeddings.npz", adapter.read_json(policy_path))
    assert mixture == adapter.read_json(mixture_path)
    assert abs(adapter.metrics(gt, mixture)["Score"] - 75.83930260989955) < 1e-10
    raw_oof = git_file(revision, "work/dino_oof_probs.csv")
    raw_test = git_file(revision, "work/dino_test_probs.csv")
    upstream_test = git_file(revision, "work/dino_final_submission.csv")
    assert sha_bytes(raw_oof) == upstream["oof_probabilities_sha256"]
    assert args.attached_test.read_bytes() == upstream_test
    args.out_dir.mkdir(parents=True, exist_ok=False)
    oof_path, test_path = [args.out_dir / f"dino_{s}_probs.csv" for s in ("oof", "test")]
    for p, raw in ((oof_path, raw_oof), (test_path, raw_test)):
        with p.open("xb") as f:
            f.write(raw)
    probabilities = adapter.read_probs(oof_path, adapter.inventory(base, fold), fold)
    dino = adapter.apply(base, probabilities, "both")
    stack = overlay_take(dino, mixture)
    raw, test = adapter.load_test()
    probabilities = adapter.read_probs(test_path, adapter.inventory(test, None), None)
    test_cells = adapter.export_cells(raw, test, adapter.apply(test, probabilities, "both"), "both")
    reader = csv.DictReader(io.StringIO(upstream_test.decode("utf-8")))
    committed_cells = {r["image_id"]: r["prediction"] for r in reader}
    assert list(test_cells) == list(committed_cells)
    score_rounding = 0
    for i, cell in test_cells.items():
        for p, q in zip(cell.split(";"), committed_cells[i].split(";"), strict=True):
            if p == q:
                continue
            a, b = p.split(), q.split()
            assert len(a) == len(b) == 10 and a[:4] + a[5:] == b[:4] + b[5:]
            assert math.isclose(float(a[4]), float(b[4]), rel_tol=1e-14, abs_tol=0.)
            score_rounding += 1

    scopes = {"pooled": gt, **{str(k): {i: gs for i, gs in gt.items() if fold[i] == k}
                              for k in sorted(set(fold.values()))}}
    report = {name: verify_components(gs, base, dino, mixture, stack) for name, gs in scopes.items()}
    for name, row in report.items():
        expected = upstream["variants"]["both"]["pooled" if name == "pooled" else "folds"]
        if name != "pooled":
            expected = expected[name]["metrics"]
        assert row["dino_both"] == expected
        print(name, *(f"{row[v]['Score']:.5f}" for v in ("base", "dino_both", "mixture", "stack", "delta_mixture")))
    adapter.write_json(args.out_dir / "report.json", dict(
        revision=revision, dino_attestation=upstream["attestation"], oof_probabilities_sha=sha_bytes(raw_oof),
        test_probabilities_sha=sha_bytes(raw_test), attached_test_sha=sha_bytes(upstream_test),
        test_replay_score_rounding_slots=score_rounding,
        mixture_sha=adapter.sha(mixture_path), mixture_policy_sha=adapter.sha(policy_path), scopes=report))
    adapter.write_json(args.out_dir / "stack_oof.json", stack)


if __name__ == "__main__":
    main()
