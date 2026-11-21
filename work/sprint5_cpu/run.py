"""Two fixed CPU A/Bs on the exact Sprint 4 gated detector/attribute baseline."""
import importlib.util
import math
from pathlib import Path


OUT = Path(__file__).resolve().parent
GATE_PATH = OUT.parent / "sprint4_volume_gate" / "run.py"
spec = importlib.util.spec_from_file_location("sprint4_volume_gate_run", GATE_PATH)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
a = gate.anchor
m = a.metric

from attribute_postprocess import DEFAULT_MODEL, DensityPrior, TakeRule

BASE_SCORE = 74.86069298423111
BASE_PUBLIC_REFERENCE = 81.7147  # historical result of this exact CSV, not re-measured
BASE_SHA = "e9c38fe9349c95fac7a6191c55fb8b936f30c7302d356d709ac666b36a65c49f"
# Fixed, no threshold search: shipped rescue setting and 2/4/8x the L960 admission floor.
RESCUE = TakeRule(.85, 300.)
PRUNE = (.05, .10, .20)


def fit_prior(train, frames, ix):
    records, obj = [], set()
    for j in ix:
        i, _, ann, o, f = train[j]
        if frames[i][0] == "real":
            v = ann["attributes"]
            records.append((o, f, v["material"], v["volume_ml"], v["mass_g"]))
            obj.add(o)
    prior = DensityPrior.fit(records)
    a.need(len(prior.entries) == len(obj), "Expected one density per object")
    return prior, dict(real_train_rows=len(records), real_train_objects=sorted(obj))


def rescue(pred, prior):
    out, hits = {}, {}
    for i, ps in pred.items():
        out[i] = list(ps)
        for j, p in enumerate(ps):
            if p["take"] == 0 and 0 < p["volume"] <= RESCUE.small_ml:
                prob = prior.light_probability(p["form"], p["material"], p["volume"])
                if prob >= RESCUE.cut:
                    out[i][j] = dict(p, take=1)
                    hits.setdefault(i, {})[j] = prob
    return out, hits


def prune(pred, n_anchor, cut):
    out, drops = {}, {}
    for i, ps in pred.items():
        drops[i] = [j for j in range(n_anchor[i], len(ps)) if ps[j]["score"] < cut]
        out[i] = [p for j, p in enumerate(ps) if j not in drops[i]]
    return out, drops


def paired_with_folds(gt, base, pred, specs):
    return dict(pooled=a.paired(gt, base, pred), folds={
        s["fold"]: a.paired({i: gt[i] for i in s["val_real_image_ids"]}, base, pred) for s in specs})


def robust(report, invariants):
    d = report["pooled"]["delta"]["Score"]
    n = sum(f["delta"]["Score"] > 1e-10 for f in report["folds"].values())
    return invariants and d > 1e-10 and n >= 3 and report["loo"]["min_delta"] > 1e-10


def attr_audit(gt, base, pred):
    by_class = {k: dict(gt=0, matched=0, base_correct=0, candidate_correct=0) for k in ("heavy", "light")}
    churn = dict(match=0, M=0, V=0, S=0)
    for i, gs in gt.items():
        old, new = base[i], pred[i]
        slots = [{id(p): j for j, p in enumerate(ps)} for ps in (old, new)]
        for g, p, q in zip(gs, m._match(gs, old), m._match(gs, new), strict=True):
            c = by_class["light" if g["take"] == 1 else "heavy"]
            c["gt"] += 1
            c["matched"] += p is not None
            c["base_correct"] += p is not None and p["take"] == g["take"]
            c["candidate_correct"] += q is not None and q["take"] == g["take"]
            # Retained prune rows are identity-preserving; rescue rows can be copied.
            if len(old) == len(new):
                churn["match"] += slots[0].get(id(p)) != slots[1].get(id(q))
            else:
                churn["match"] += p is not q
            for k, attr in (("M", "material"), ("V", "volume"), ("S", "take")):
                churn[k] += (p or {}).get(attr) != (q or {}).get(attr)
    for c in by_class.values():
        for name in ("base", "candidate"):
            c[f"{name}_accuracy"] = c[f"{name}_correct"] / c["gt"] if c["gt"] else None
    return dict(by_mass=by_class, match_churn=churn)


def activation_audit(gt, old_rank, base, hits):
    rows = []
    for i in gt:
        ps, js = base[i], hits.get(i, {})
        labels = {id(p): ("light" if g["take"] else "heavy")
                  for p, g in zip(m._match(gt[i], ps), gt[i], strict=True) if p is not None}
        for j, prob in js.items():
            p = ps[j]
            rows.append(dict(image_id=i, slot=j, probability=prob, volume=p["volume"],
                             volume_changed=p["volume"] != old_rank[i][j]["volume"],
                             matched=labels.get(id(p), "unmatched")))
    return dict(eligible=sum(p["take"] == 0 and 0 < p["volume"] <= RESCUE.small_ml
                            for i in gt for p in base[i]),
                rows=len(rows), frames=len({r["image_id"] for r in rows}),
                volume_changed=sum(r["volume_changed"] for r in rows),
                by_match={k: sum(r["matched"] == k for r in rows) for k in ("heavy", "light", "unmatched")},
                events=rows)


def ap_drop_audit(gt, base, drops):
    """Trace the official stable per-class greedy order; labels are diagnostics only."""
    result = {}
    for form in m.FORMS:
        dets = [(i, j, p) for i in gt for j, p in enumerate(base[i]) if p["form"] == form]
        dets.sort(key=lambda e: -e[2]["score"])
        used = {i: set() for i in gt}
        tp, dropped = [], []
        for rank, (i, j, p) in enumerate(dets, 1):
            best, best_j = 0., None
            for k, g in enumerate(gt[i]):
                if g["form"] == form and k not in used[i]:
                    iou = m._iou(p["box"], g["box"])
                    if iou > best:
                        best, best_j = iou, k
            if best >= m.IOU_THR:
                used[i].add(best_j)
                tp.append(rank)
            if j in drops[i]:
                dropped.append(dict(image_id=i, slot=j, rank=rank, score=p["score"], tp=best >= m.IOU_THR))
        tail = max(tp, default=0)
        result[form] = dict(last_tp_rank=tail, removed_tp=sum(d["tp"] for d in dropped),
                            removed_fp_before_last_tp=sum(not d["tp"] and d["rank"] < tail for d in dropped),
                            removed_fp_after_last_tp=sum(not d["tp"] and d["rank"] > tail for d in dropped),
                            events=dropped)
    return result


def main():
    policy = a.read(DEFAULT_MODEL)
    a.need(policy["rescue_small"] == vars(RESCUE) and
           policy["prior"]["temperature"] == DensityPrior.__dataclass_fields__["temperature"].default,
           "Shipped rescue configuration changed")
    camera_p = a.DATA / "camera.json"
    camera = a.read(camera_p)
    a.need(camera["depth_unit"] == "mm" and camera["depth_aligned_to"] == "color", "Bad camera domain")
    specs = a.read(a.MANIFEST)["folds"]
    frames, gt, objects = a.labelled()
    hashes = {}
    for p, expected in a.EXPECTED.items():
        a.need(a.sha(p) == expected, f"Frozen input changed: {p}")
    frozen, omap, vs, _, _, tmap, ts = a.prepare_sources(frames, specs, hashes)
    oof_counts = a.read(a.CACHE / "predictions.json")["n640_rescue"]
    test_anchors = {i: m._parse(s, True, i) for i, s in a.csv_rows(a.ANCHOR_TEST).items()}
    rank_report_p = a.OUT / "report.json"
    rank_report = a.read(rank_report_p)
    oof_path, rank_csv = a.OUT / "selected_oof.json", a.OUT / "submission_selected.csv"
    for p, k in ((oof_path, "oof"), (rank_csv, "csv")):
        a.need(a.sha(p) == rank_report["scores"][k], f"Rank input SHA changed: {p}")
    ranked = a.read(oof_path)
    a.churn(gt, frozen, ranked)
    for i, old in oof_counts.items():
        a.need(all(vs[k][1] == j and a.same_detection(p, vs[k][3])
                   for j, (p, k) in enumerate(zip(old, omap[i][1][:len(old)], strict=True))), i)
    omap = {i: keys for i, (_, keys) in omap.items()}
    gate.audit_mapping(ranked, omap, vs)
    base_csv = GATE_PATH.parent / "submission_best_verified.csv"
    gate_report_p = GATE_PATH.parent / "report.json"
    gate_report = a.read(gate_report_p)
    a.need(gate_report["selected"] == gate.GATED and gate_report["config"] == gate.CONFIG and
           a.sha(base_csv) == gate_report["output_sha256"] == BASE_SHA, "Unexpected exact base CSV")
    exact_test = a.csv_rows(base_csv)
    ranked_rows = a.csv_rows(rank_csv)
    test_rank = {i: m._parse(s, True, i) for i, s in ranked_rows.items()}
    a.token_changes(ranked_rows, exact_test, {7})
    gate.audit_mapping(test_rank, tmap, ts)
    frozen_test_rows = a.csv_rows(a.BASE_CSV)
    for i, ps in test_anchors.items():
        freeze = m._parse(frozen_test_rows[i], True, i)
        a.need(all(a.same_detection(p, q) for p, q in zip(ps, freeze[:len(ps)], strict=True)), i)
    train = a.training_records(frames)
    geo = gate.Geometry(camera)
    gtrain = {j: geo.gt(r) for j, r in enumerate(train) if r[4] == "bulk" and frames[r[0]][0] == "real"}
    vg = {j: geo.predicted(src, "real_oof") for j, src in enumerate(vs) if src[3]["form"] == "bulk"}
    tg = {j: geo.predicted(src, "real_test") for j, src in enumerate(ts) if src[3]["form"] == "bulk"}
    a.need(bool(geo.domains["real_test"]) and geo.domains["real_test"] <= geo.domains["real_oof"], "Test domain")
    print(f"Gate geometry: {len(gtrain)} GT, {len(vg)} OOF, {len(tg)} test", flush=True)
    base, rescued, hits, density = {}, {}, {}, {}
    for s in specs:
        k = s["fold"]
        ix = a.fit_rows(train, s)
        fg = {i: gt[i] for i in s["val_real_image_ids"]}
        scale = gate.fit_scale(train, gtrain, ix)
        a.need(math.isclose(scale["log_scale"], gate_report["train_scales"][str(k)]["log_scale"], abs_tol=1e-12),
               f"Gate scale differs in fold {k}")
        fold = {i: gate.update(ranked[i], omap[i], vg, scale["log_scale"], vs) for i in fg}
        a.need(not set(fold) & set(base), "Overlapping outer folds")
        base.update(fold)
        measured = a.metrics(fg, fold)
        a.need(all(math.isclose(measured[mv], gate_report["folds"][str(k)]["candidate"][mv],
                                rel_tol=0, abs_tol=1e-10) for mv in ("Score", "D", "M", "V", "S")), k)
        prior, density[k] = fit_prior(train, frames, ix)
        a.need(not set(density[k]["real_train_objects"]) & {o for i in fg for o in objects[i]}, k)
        p, h = rescue(fold, prior)
        rescued.update(p)
        hits.update(h)
    a.need(set(base) == set(gt) and abs(a.metrics(gt, {i: base[i] for i in gt})["Score"] - BASE_SCORE) < 1e-10,
           "Wrong gated OOF baseline")
    a.dump(OUT / "base_7486069_oof.json", {i: base[i] for i in gt})
    baseline_mass = attr_audit(gt, base, base)["by_mass"]
    a.need([(baseline_mass[k]["base_correct"], baseline_mass[k]["gt"], baseline_mass[k]["matched"])
            for k in ("heavy", "light")] == [(38, 44, 44), (105, 170, 170)], "Wrong baseline take/match counts")
    print("Baseline heavy/light:", {k: (v["base_correct"], v["gt"], v["matched"])
                                     for k, v in baseline_mass.items()}, flush=True)
    a.need(a.test_counts(exact_test) == (137, 342, 10), "Wrong base CSV inventory")
    test_scale = gate.fit_scale(train, gtrain, range(len(train)))
    rebuilt = {i: a.replace_tokens(ranked_rows[i], gate.update(test_rank[i], tmap[i], tg, test_scale["log_scale"], ts),
                                   {7: "volume"}) for i in ranked_rows}
    a.token_changes(exact_test, rebuilt, {7})
    a.need(all(math.isclose(float(p.split()[7]), float(q.split()[7]), rel_tol=1e-12, abs_tol=0.)
               for i in exact_test if exact_test[i] != "none"
               for p, q in zip(exact_test[i].split(";"), rebuilt[i].split(";"), strict=True)),
           "Rebuilt test gate has different volume")
    a.dump(OUT / "rescue_small_oof.json", {i: rescued[i] for i in gt})

    splits = {s["fold"]: {i: gt[i] for i in s["val_real_image_ids"]} for s in specs}
    report_r = paired_with_folds(gt, base, rescued, specs)
    report_r.update(loo=a.removal(gt, objects, base, rescued),
                    audit=attr_audit(gt, base, rescued),
                    activations=activation_audit(gt, ranked, base, hits))
    for k, g in splits.items():
        report_r["folds"][k].update(audit=attr_audit(g, base, rescued),
                                     activations=activation_audit(g, ranked, base, hits))
    keep = all(d["delta"][k] == 0 for d in (report_r["pooled"], *report_r["folds"].values()) for k in ("D", "M", "V"))
    a.need(keep and not any(report_r["audit"]["match_churn"][k] for k in ("match", "M", "V")),
           "Rescue changed more than take")
    report_r["robust"] = robust(report_r, keep)
    summarize("rescue_small", report_r)

    reports, cuts = {}, {}
    for cut in PRUNE:
        variant = f"prune_{cut:.2f}"
        pred, drops = prune(base, {i: len(ps) for i, ps in oof_counts.items()}, cut)
        a.dump(OUT / (variant + "_oof.json"), {i: pred[i] for i in gt})
        rep = paired_with_folds(gt, base, pred, specs)
        rep.update(loo=a.removal(gt, objects, base, pred), audit=attr_audit(gt, base, pred),
                   dropped_rows=sum(map(len, drops.values())), dropped_frames=sum(bool(d) for d in drops.values()),
                   ap_drops=ap_drop_audit(gt, base, drops))
        for k, g in splits.items():
            rep["folds"][k].update(audit=attr_audit(g, base, pred),
                                    dropped_rows=sum(len(drops[i]) for i in g),
                                    dropped_frames=sum(bool(drops[i]) for i in g),
                                    ap_drops=ap_drop_audit(g, base, drops))
        keep = (all(d["delta"][k] == 0 for d in (rep["pooled"], *rep["folds"].values()) for k in ("M", "V", "S"))
                and not any(rep["audit"]["match_churn"].values()))
        rep["robust"] = robust(rep, keep) and rep["pooled"]["delta"]["D"] > 1e-10
        reports[variant], cuts[variant] = rep, cut
        summarize(variant, rep)

    selected_prune = max((v for v in reports if reports[v]["robust"]),
                         key=lambda v: reports[v]["pooled"]["delta"]["Score"], default=None)
    selected_rescue = report_r["robust"]
    combined, export = None, None
    if selected_rescue or selected_prune is not None:
        chosen = rescued if selected_rescue else base
        if selected_prune is not None:
            chosen, _ = prune(chosen, {i: len(ps) for i, ps in oof_counts.items()}, cuts[selected_prune])
        combined = paired_with_folds(gt, base, chosen, specs)
        combined.update(loo=a.removal(gt, objects, base, chosen), audit=attr_audit(gt, base, chosen))
        preserved = ("M", "V") if selected_rescue else ("M", "V", "S")
        keep = all(d["delta"][k] == 0 for d in (combined["pooled"], *combined["folds"].values()) for k in preserved)
        combined["robust"] = robust(combined, keep)
        a.dump(OUT / "selected_oof.json", {i: chosen[i] for i in gt})
        if combined["robust"]:
            export = export_test(exact_test, test_anchors, train, frames, selected_rescue, cuts.get(selected_prune))

    for p in (Path(__file__).resolve(), GATE_PATH, Path(a.__file__), a.ROOT / "score.py",
              a.ROOT / "fusion/attribute_postprocess.py", DEFAULT_MODEL, camera_p,
              a.DATA / "labels/real_train.json", a.DATA / "labels/synth_train.json", a.MANIFEST,
              a.CACHE / "predictions.json", a.CACHE / "mask_index.json", a.FORM_TEST, a.TEST_RAW_CSV,
              a.WBF / "base_masks.npz", a.WBF / "base_detections.jsonl", *a.EXPECTED,
              rank_report_p, oof_path, rank_csv, base_csv, gate_report_p):
        hashes[p] = a.sha(p)
    a.dump(OUT / "report.json", dict(
        baseline=dict(verified_oof_score=BASE_SCORE, historic_public=BASE_PUBLIC_REFERENCE, test_csv_sha256=BASE_SHA,
                      reproduced_test_prediction_text_exact=rebuilt == exact_test, test_counts=a.test_counts(exact_test),
                      by_mass=baseline_mass),
        config=dict(rescue=vars(RESCUE), prune_confidences=PRUNE, prune_uses_final_reranked_confidence=True,
                    loo="remove all real OOF frames containing object; fixed predictions, no refitting",
                    promotion="pooled gain > 0; >=3 improving folds; min object-removal gain > 0; metric/match checks"),
        density_fit=density, rescue_small=report_r, prune=reports, selected_rescue=selected_rescue,
        selected_prune=selected_prune, recombined=combined, test_export=export,
        input_sha256={str(p.relative_to(a.ROOT)): h for p, h in hashes.items()}))
    print("Selected:", selected_rescue, selected_prune, "export:", export, flush=True)


def summarize(name, report):
    print(name, report["pooled"]["delta"], "folds", [round(r["delta"]["Score"], 6) for r in report["folds"].values()],
          "min LOO", report["loo"]["min_delta"], "robust", report["robust"], flush=True)


def export_test(rows, anchors, train, frames, do_rescue, cut):
    base = {i: m._parse(s, True, i) for i, s in rows.items()}
    hits = {}
    if do_rescue:
        prior, _ = fit_prior(train, frames, range(len(train)))
        _, hits = rescue(base, prior)
    if cut is None:
        drops = {i: [] for i in rows}
    else:
        _, drops = prune(base, {i: len(ps) for i, ps in anchors.items()}, cut)
    output = {}
    for i, text in rows.items():
        pieces = []
        for j, part in enumerate([] if text == "none" else text.split(";")):
            if j in drops[i]:
                a.need(j >= len(anchors[i]), "Attempted to prune test anchor")
            elif j in hits.get(i, {}):
                left, value = part.rsplit(maxsplit=1)
                a.need(int(float(value)) == 0, "Attempted to rescue non-zero take")
                pieces.append(left + " 1")
            else:
                pieces.append(part)
        output[i] = ";".join(pieces) or "none"
    path = OUT / "submission_sprint5_cpu.csv"
    if output == rows:
        return dict(path=None, same_as_exact_base=True, sha256=BASE_SHA, counts=a.test_counts(output),
                    take_edits=0, removed_rows=0)
    a.write_rows(path, output)
    a.need(a.csv_rows(path) == output, "Test export round trip")
    a.validate_csv(path)
    return dict(path=str(path.relative_to(a.ROOT)), sha256=a.sha(path), counts=a.test_counts(output),
                take_edits=sum(j not in drops[i] for i, js in hits.items() for j in js),
                removed_rows=sum(map(len, drops.values())))


if __name__ == "__main__":
    gate.cv2.setNumThreads(1)
    with gate.threadpool_limits(2):
        main()
