"""Grouped OOF A/B of visual instance consensus and material-mixture take."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from fusion import sprint5_crop_adapter as adapter
from fusion.attribute_postprocess import DensityPrior
import score as official


# Fixed before seeing outer results. Complete-link prevents single-edge chaining.
CLUSTER = dict(dino_weight=.65, area_penalty=.035, aspect_penalty=.05, min_affinity=.78)


def cluster_ids(images, boxes, areas, dino, clip):
    """One batch (one validation fold or all test); no IDs enter the affinity."""
    if len(images) < 2:
        return np.ones(len(images), dtype=int)
    w, h = (np.asarray(boxes)[:, 2:] - np.asarray(boxes)[:, :2]).T
    log_area = np.log(areas)
    log_aspect = np.log(w / h)
    d, c = normalize(dino), normalize(clip)
    a = CLUSTER["dino_weight"]
    affine = a * (d @ d.T) + (1 - a) * (c @ c.T)
    affine -= CLUSTER["area_penalty"] * np.abs(log_area[:, None] - log_area)
    affine -= CLUSTER["aspect_penalty"] * np.abs(log_aspect[:, None] - log_aspect)
    distance = 1 - np.clip(affine, -1, 1)
    # Two slots in one image must not vote as separate views of a cluster.
    same_frame = np.equal.outer(images, images)
    distance[same_frame] = 3.
    np.fill_diagonal(distance, 0.)
    tree = linkage(squareform(distance, checks=False), method="complete")
    return fcluster(tree, 1 - CLUSTER["min_affinity"], criterion="distance")


def consensus(rows, posteriors, labels, groups):
    """Soft material vote, median volume and weighted majority of frozen take."""
    result = [dict(d) for d in rows]
    for group in np.unique(groups):
        ix = np.flatnonzero(groups == group)
        if len(ix) < 2:
            continue
        weights = np.array([rows[j]["score"] for j in ix])
        p = np.average(posteriors[ix], axis=0, weights=weights)
        volume = float(np.median([rows[j]["volume"] for j in ix]))
        take_vote = np.average([rows[j]["take"] for j in ix], weights=weights)
        take = int(take_vote > .5) if take_vote != .5 else rows[ix[weights.argmax()]]["take"]
        for j in ix:
            result[j].update(material=str(labels[p.argmax()]), volume=volume, take=take)
    return result


def light_scores(forms, volumes, posteriors, labels, prior):
    """P(light|crop) and its MAP-material-only control using the same density CDF."""
    cdf = np.asarray([[prior.light_probability(f, m, v) for m in labels]
                      for f, v in zip(forms, volumes, strict=True)], dtype=float).reshape(-1, len(labels))
    # The second arm is a control, not the frozen take classifier.
    return (cdf * posteriors).sum(axis=1), cdf[np.arange(len(cdf)), posteriors.argmax(axis=1)]


def density_prior(data):
    cols = ["object_id", "form", "material", "volume_ml", "mass_g"]
    return DensityPrior.fit(data.loc[data.source.eq("real"), cols].itertuples(index=False, name=None))


def choose_cut(prob, take, objects):
    """Optimize balanced S exactly on inner cross-fit rows, no outer predictions."""
    assert set(take) == {0, 1} and np.isfinite(prob).all()
    xs = np.unique(np.r_[0., prob, 1.])
    cuts = np.unique(np.r_[0., .5, 1., (xs[:-1] + xs[1:]) / 2])
    hits = (prob[:, None] >= cuts) == take[:, None]

    def score(keep):
        return np.mean([hits[keep & (take == t)].mean(axis=0) for t in (0, 1)
                        if np.any(keep & (take == t))], axis=0)

    s = score(np.ones(len(take), dtype=bool))
    robust = np.min([score(objects != o) for o in np.unique(objects)], axis=0)
    j = max(range(len(cuts)), key=lambda j: (round(s[j], 12), round(robust[j], 12), -abs(cuts[j] - .5)))
    return float(cuts[j]), float(s[j])


def material_prob(models, x):
    return sum(models[j % 2].predict_proba(a) for j, a in enumerate(x)) / len(x)


def fit_material(train, xy, labels):
    models = [LogisticRegression(C=1, class_weight="balanced", max_iter=1000)
              .fit(v[train.index], train.material) for v in xy[:2]]
    assert all(np.array_equal(m.classes_, labels) for m in models)
    return models


def policy_params(models, labels, prior, cut):
    return dict(labels=list(labels), cut=cut, density_entries=prior.entries, temperature=prior.temperature,
                heads=[dict(coef=m.coef_.tolist(), intercept=m.intercept_.tolist()) for m in models])


def apply_mixture(rows, views, policy):
    """Targeted B handoff: final detector rows and normalized DINO/CLIP orig+flip per slot.

    `rows` and all four arrays must already be in the same slot order. Use the
    corresponding fold policy for a validation batch; use 'test' after full fit.
    Returns copies, changing only take (legacy take_p is not a mixture probability).
    """
    if len(views) != 4 or any(len(v) != len(rows) or not np.isfinite(v).all() for v in views):
        raise ValueError("Expected four finite, slot-aligned frozen encoder arrays")
    if not rows:
        return []
    posteriors = []
    for j, x in enumerate(views):
        h = policy["heads"][j % 2]
        posteriors.append(softmax(normalize(x) @ np.asarray(h["coef"]).T + h["intercept"], axis=1))
    p = np.mean(posteriors, axis=0)
    prior = DensityPrior(tuple(policy["density_entries"]), policy["temperature"])
    mix, _ = light_scores([d["form"] for d in rows], [d["volume"] for d in rows], p, policy["labels"], prior)
    return [dict(d, take=int(s >= policy["cut"])) for d, s in zip(rows, mix, strict=True)]


def fold_calibration(train, xy, joint):
    """Generate fresh inner predictions; rest-of-outer OOF would leak upstream."""
    records = []
    labels = np.asarray(joint.labels)
    for fit, hold in joint.grouped_folds(train, 3):
        hold = hold.loc[hold.source.eq("real")]
        xx = [v[hold.index] for v in xy]
        model = joint.fit_up(fit, xy)
        p = material_prob(model["mat"], xx)
        volume = joint.up(model, hold, xx).volume_pred_ml
        mix, hard = light_scores(hold.form, volume, p, labels, density_prior(fit))
        records.append(hold[["object_id", "take"]].assign(mixture=mix, map_density=hard))
    rows = pd.concat(records, verify_integrity=True)
    assert set(rows.index) == set(train.index[train.source.eq("real")])
    return {v: dict(zip(("cut", "inner_S"), choose_cut(
        rows[v].to_numpy(), rows["take"].to_numpy(), rows.object_id.to_numpy()), strict=True))
            for v in ("mixture", "map_density")}


def audit(gt, base, pred, intended):
    edits, matched_edits = Counter(), Counter()
    hits = {t: Counter(gt=0, before=0, after=0) for t in (0, 1)}
    for i, gs in gt.items():
        before, after = adapter.slots(gs, base[i]), adapter.slots(gs, pred[i])
        assert before == after
        for j, (p, q) in enumerate(zip(base[i], pred[i], strict=True)):
            changed = {k for k in p if p[k] != q[k]}
            assert changed <= intended and p.keys() == q.keys()
            edits.update(changed)
            if j in before:
                matched_edits.update(changed)
        for g, j in zip(gs, before, strict=True):
            t = g["take"]
            hits[t]["gt"] += 1
            hits[t]["before"] += j is not None and base[i][j]["take"] == t
            hits[t]["after"] += j is not None and pred[i][j]["take"] == t
    return dict(edits=edits, matched_edits=matched_edits, take_hits=hits)


def run_ab(embeddings_path, fit_test_policy=False):
    import astra_joint04 as joint

    folds, real, _, provenance, limits = adapter.training_contract()
    gt, base = adapter.load_oof(adapter.OOF_BASE, folds)
    assert len(set(folds.values())) == 5 and len(set().union(*real.values())) == 40
    assert all(len({folds[i] for i in real if obj in real[i]}) == 1 for obj in set().union(*real.values()))
    inv = adapter.inventory(base, folds)
    keys = [(i, str(j), sha) for i, j, _, sha, *_ in inv]
    with np.load(embeddings_path, allow_pickle=False) as z:
        assert z["keys"].tolist() == [list(k) for k in keys]
        index = z["donor_index"]
        assert index.shape == (len(keys),) and index.min() == 0 and index.max() < len(z["area"])
        vx = [z[k][index] for k in ("d", "c", "df", "cf")]
        area = z["area"][index]
    assert all(np.isfinite(a).all() for a in vx) and np.all(area > 0)
    data, _ = joint.read_data()
    data = data.merge(pd.read_csv(joint.GEOMETRY_PATH), on=["image_id", "ann_id"],
                      validate="one_to_one").assign(split="train")
    train_keys = joint.keys(data)
    with np.load(joint.W / "astra_joint04_train_flip.npz", allow_pickle=False) as flip:
        assert flip["ids"].tolist() == train_keys
        xy = [normalize(joint.embeddings(k, train_keys)) for k in ("astra_", "astra_clip_")]
        xy.extend([flip["d"], flip["c"]])
    variants = {v: {} for v in ("cluster", "mixture", "map_density")}
    metadata = {}
    cluster_by_slot = {}
    labels = np.asarray(joint.labels)
    objects = adapter.ordered_gt_objects(gt)
    policies = {}
    for k in sorted(set(folds.values())):
        scope = limits[str(k)]
        train = data.loc[data.image_id.isin(scope["real"] + scope["allowed_synth"])]
        assert not set(train.object_id) & set(scope["forbidden_object_ids"])
        assert not set(train.image_id) & {i for i in folds if folds[i] == k}
        fit_rows = [j for j, x in enumerate(inv) if x[2] == k]
        slots = [(inv[j][0], inv[j][1]) for j in fit_rows]
        rows = [base[i][j] for i, j in slots]
        x = [v[fit_rows] for v in vx]
        calibration = fold_calibration(train, xy, joint)
        ms = fit_material(train, xy, labels)
        p = material_prob(ms, x)
        assert np.isfinite(p).all() and np.allclose(p.sum(axis=1), 1.)
        groups = cluster_ids([i for i, _ in slots], [d["box"] for d in rows], area[fit_rows],
                             normalize(x[0] + x[2]), normalize(x[1] + x[3]))
        clustered = consensus(rows, p, labels, groups)
        prior = density_prior(train)
        mix, hard = light_scores([d["form"] for d in rows], [d["volume"] for d in rows], p, labels, prior)
        output = dict(cluster=clustered, mixture=[dict(d, take=int(s >= calibration["mixture"]["cut"]))
                                                   for d, s in zip(rows, mix, strict=True)],
                      map_density=[dict(d, take=int(s >= calibration["map_density"]["cut"]))
                                   for d, s in zip(rows, hard, strict=True)])
        policies[str(k)] = policy_params(ms, labels, prior, calibration["mixture"]["cut"])
        assert output["mixture"] == apply_mixture(rows, x, policies[str(k)])
        images = [i for i in base if folds[i] == k]
        for v, preds in output.items():
            variants[v].update({i: [] for i in images})
            for (i, j), d in zip(slots, preds, strict=True):
                assert len(variants[v][i]) == j
                variants[v][i].append(d)
        sizes = Counter(groups)
        cluster_by_slot.update({slot: (k, int(g)) for slot, g in zip(slots, groups, strict=True)})
        metadata[k] = dict(calibration=calibration, train_real=int(train.source.eq("real").sum()),
                           train_synth=int(train.source.eq("synth").sum()), frames=len(images),
                           clusters=sum(s > 1 for s in sizes.values()), clustered_slots=sum(s for s in sizes.values() if s > 1),
                           posterior_argmax_equals_frozen=int(np.sum(labels[p.argmax(axis=1)] ==
                                                                      [d["material"] for d in rows])),
                           slots=len(rows))
        print(f"Fold {k}: {metadata[k]}", flush=True)

    # GT is only used for evaluation from here, not as a clustering input.
    matched_groups = {}
    for i, gs in gt.items():
        for j, obj in zip(adapter.slots(gs, base[i]), objects[i], strict=True):
            if j is not None:
                matched_groups.setdefault(cluster_by_slot[i, j], []).append(obj)
    for k in metadata:
        pairs = [(a, b) for (fold, _), os in matched_groups.items() if k == fold
                 for j, a in enumerate(os) for b in os[j+1:]]
        metadata[k]["same_object_pairs"] = sum(a == b for a, b in pairs)
        metadata[k]["matched_pairs"] = len(pairs)
        metadata[k]["mixed_matched_clusters"] = sum(k == f and len(set(os)) > 1
                                                     for (f, _), os in matched_groups.items())
    scopes = {"pooled": gt, **{str(k): {i: gs for i, gs in gt.items() if folds[i] == k} for k in metadata}}
    result = {"baseline": {name: adapter.metrics(scope, base) for name, scope in scopes.items()}}
    for v, pred in variants.items():
        fields = {"material", "volume", "take"} if v == "cluster" else {"take"}
        result[v] = {}
        for name, scope in scopes.items():
            b = result["baseline"][name]
            m = adapter.metrics(scope, pred)
            assert m["D"] == b["D"] and (v == "cluster" or (m["M"] == b["M"] and m["V"] == b["V"]))
            result[v][name] = dict(metrics=m, delta=adapter.delta(b, m), audit=audit(scope, base, pred, fields))
        loo = {}
        for obj in sorted(set().union(*real.values())):
            scope = {i: gs for i, gs in gt.items() if obj not in real[i]}
            loo[obj] = adapter.metrics(scope, pred)["Score"] - adapter.metrics(scope, base)["Score"]
        wins = sum(result[v][str(k)]["delta"]["Score"] > 1e-9 for k in metadata)
        result[v]["robustness"] = dict(positive_folds=wins, min_loo_delta=min(loo.values()), loo=loo,
                                       gate=(wins >= 3 and min(loo.values()) > 1e-9
                                             and result[v]["pooled"]["delta"]["Score"] > 1e-9))
    if fit_test_policy:
        scope = limits["test"]
        train = data.loc[data.image_id.isin(scope["real"] + scope["allowed_synth"])]
        test_cut = fold_calibration(train, xy, joint)["mixture"]["cut"]
        policies["test"] = policy_params(fit_material(train, xy, labels), labels, density_prior(train), test_cut)
        print(f"Full-train deployment cut (not an OOF setting): {test_cut:.8f}", flush=True)
    return (dict(cluster_config=CLUSTER, inputs=dict(provenance, baseline_sha=adapter.OOF_CANONICAL,
                                                    embeddings_sha=adapter.sha(embeddings_path)),
                 folds=metadata, results=result), variants, policies)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path)
    parser.add_argument("--policy-out", type=Path, help="Save five fold policies plus a separate full-train fit")
    args = parser.parse_args()
    if any(p.exists() for p in (args.report, args.predictions_dir, args.policy_out) if p is not None):
        raise FileExistsError("Refusing to overwrite output")
    with threadpool_limits(4):
        report, variants, policies = run_ab(args.embeddings, args.policy_out is not None)
    adapter.write_json(args.report, report)
    if args.policy_out is not None:
        adapter.write_json(args.policy_out, dict(version=1, folds=policies, inputs=report["inputs"]))
    if args.predictions_dir is not None:
        args.predictions_dir.mkdir()
        for v, p in variants.items():
            adapter.write_json(args.predictions_dir / f"{v}.json", p)
    results = report["results"]
    print("fold     baseline     cluster Δ    mixture Δ        MAP Δ")
    for k, b in results["baseline"].items():
        print(f"{k:6} {b['Score']:10.5f}" + "".join(
            f" {results[v][k]['delta']['Score']:+12.5f}" for v in variants))


if __name__ == "__main__":
    main()
