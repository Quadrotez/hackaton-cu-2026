"""Verify original rescue provenance, then reapply its unchanged rules to gated volumes."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

spec = importlib.util.spec_from_file_location("sprint5_cpu_run", Path(__file__).with_name("run.py"))
sys.modules[spec.name] = cpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cpu)
from sprint5_cpu_run import (BASE_SCORE, BASE_SHA, DEFAULT_MODEL, GATE_PATH, OUT, TakeRule, a, attr_audit,
                            fit_prior, m, paired_with_folds, robust, summarize)
from attribute_postprocess import AttributePolicy
from evaluate_attributes import rescue_selection


def prior_agrees(p, q):
    return p.temperature == q.temperature and tuple(map(tuple, p.entries)) == tuple(map(tuple, q.entries))


def gated_replay(base, frozen, raw, old, policy):
    out, details = {}, {}
    a.need(set(base) == set(frozen) == set(raw) == set(old), "Frame coverage mismatch")
    for i in base:
        front = len(raw[i])
        a.need(len(old[i]) == front and len(base[i]) == len(frozen[i]) >= front, (i, "lengths"))
        for j, (p, q) in enumerate(zip(raw[i], old[i], strict=True)):
            v, mass, t = policy.apply("rescue-small", p["form"], p["material"], p["volume"],
                                      p.get("mass", 0.), p["take"])
            a.need(dict(p, volume=v, **({"mass": mass} if "mass" in p else {}), take=t) == q,
                   (i, j, "Historical rescue fails exact replay"))
            a.need(a.same_detection(frozen[i][j], q), (i, j, "Anchor not aligned"))
            a.need(not p["take"] or q["take"] == 1, (i, j, "CatBoost positive lost"))
        out[i], details[i] = [], []
        for j, (p, q) in enumerate(zip(base[i], frozen[i], strict=True)):
            a.need(tuple(p["box"]) == tuple(q["box"]) and
                   all(p[k] == q[k] for k in ("form", "take", "material")), (i, j, "Base slot not aligned"))
            donor = j if j < front else a.donor(q, old[i])
            d, o, b = raw[i][donor], old[i][donor], base[i][donor]
            a.need(all(p[k] == b[k] for k in ("material", "take", "volume")) and
                   all(q[k] == o[k] for k in ("material", "take")), (i, j, "Donor attributes not inherited"))
            a.need(j < front or d["form"] != p["form"], (i, j, "Not an alternate"))
            _, _, t = policy.apply("rescue-small", d["form"], b["material"], b["volume"], d.get("mass", 0.), d["take"])
            a.need(not d["take"] or t == 1, (i, j, "CatBoost positive altered"))
            a.need(t >= p["take"] or (d["take"], o["take"]) == (0, 1), (i, j, "Demoted non-rescue"))
            out[i].append(dict(p, take=t))
            if p["take"] != t:
                details[i].append(dict(slot=j, donor=donor, old_take=p["take"], new_take=t,
                                       donor_raw_take=d["take"], donor_old_volume=d["volume"],
                                       donor_gated_volume=b["volume"], is_alternate=j >= front))
    return out, details


def event_summary(events):
    emp = [e for es in events.values() for e in es]
    return dict(frames=sum(bool(es) for es in events.values()), rows=len(emp),
                up=sum(e["new_take"] == 1 for e in emp), down=sum(e["new_take"] == 0 for e in emp),
                alternates=sum(e["is_alternate"] for e in emp),
                events={i: es for i, es in events.items() if es})


def main():
    base_path = OUT / "base_7486069_oof.json"
    raw_path = a.WORK / "astra_joint04_oof.json"
    fit_report_path = DEFAULT_MODEL.parent / "report.json"
    inner_path = DEFAULT_MODEL.parent / "inner_oof.csv"
    cache_path = a.CACHE / "predictions.json"
    cache_report_path = a.CACHE / "report.json"
    fits, cache_fits = a.read(fit_report_path), a.read(cache_report_path)
    a.need(fits["fold_manifest_sha256"] == cache_fits["fold_manifest_sha256"] == a.sha(a.MANIFEST) and
           fits["joint04_oof_sha256"] == cache_fits["joint04_oof_sha256"] == a.sha(raw_path),
           "Original provenance mismatch")
    inner = m.pd.read_csv(inner_path)
    specs, (frames, gt, objects) = a.read(a.MANIFEST)["folds"], a.labelled()
    train = a.training_records(frames)
    deployed = AttributePolicy.load(DEFAULT_MODEL)
    base, raw, frozen = map(a.read, (base_path, raw_path, a.BASE_OOF))
    a.need(a.sha(a.BASE_OOF) == a.EXPECTED[a.BASE_OOF], "Unexpected frozen OOF")
    old = a.read(cache_path)["n640_rescue"]
    a.need(abs(a.metrics(gt, base)["Score"] - BASE_SCORE) < 1e-10, "Unexpected gated baseline")
    a.need(raw == a.read(cache_path)["n640"], "Different source in cached OOF")
    pred, events, old_anchor_flips, rule_by_fold = {}, {}, {}, {}
    for s in specs:
        k, ids = s["fold"], s["val_real_image_ids"]
        ix = a.fit_rows(train, s)
        p, _ = fit_prior(train, frames, ix)
        fit_ids = {train[j][0] for j in ix if frames[train[j][0]][0] == "real"}
        fit_objs = {train[j][3] for j in ix if frames[train[j][0]][0] == "real"}
        rows = inner.loc[inner.outer_fold.eq(k)]
        a.need(set(rows.image_id) == fit_ids and set(rows.object_id) == fit_objs and
               not set(rows.object_id) & set(s["val_object_ids"]), (k, "Inner outer-train scope"))
        rule = rescue_selection(rows)
        a.need(rule == TakeRule(**next(f["rescue_small"] for f in fits["folds"] if f["fold"] == k)) ==
               TakeRule(**next(f["rescue_rule"] for f in cache_fits["folds"] if f["fold"] == k)),
               (k, "Historic outer-train rule mismatch"))
        rule_by_fold[k] = vars(rule)
        pp, ee = gated_replay(*({i: source[i] for i in ids} for source in (base, frozen, raw, old)),
                              replace(deployed, prior=p, rescue_small=rule))
        old_anchor_flips[k] = sum(p["take"] != q["take"] for i in ids for p, q in zip(raw[i], old[i], strict=True))
        pred.update(pp)
        events.update(ee)

    a.dump(OUT / "reapply_rescue_oof.json", {i: pred[i] for i in gt})
    rep = paired_with_folds(gt, base, pred, specs)
    rep.update(loo=a.removal(gt, objects, base, pred), audit=attr_audit(gt, base, pred),
               events=event_summary(events), historic_outer_rules=rule_by_fold,
               historic_oof_anchor_flips=old_anchor_flips)
    for s in specs:
        k = s["fold"]
        g = {i: gt[i] for i in s["val_real_image_ids"]}
        rep["folds"][k].update(audit=attr_audit(g, base, pred), events=event_summary({i: events[i] for i in g}))
    invariants = all(d["delta"][k] == 0 for d in (rep["pooled"], *rep["folds"].values()) for k in ("D", "M", "V"))
    a.need(invariants and not any(rep["audit"]["match_churn"][k] for k in ("match", "M", "V")), "Only take can change")
    rep["robust"] = robust(rep, invariants)
    summarize("reapply_rescue", rep)

    raw_csv = a.ANCHOR_TEST.with_name("submission_fused_max_7897.csv")
    exact_csv = GATE_PATH.parent / "submission_best_verified.csv"
    a.need(a.sha(exact_csv) == BASE_SHA and a.sha(a.ANCHOR_TEST) == a.EXPECTED[a.ANCHOR_TEST], "Exact CSV mismatch")
    old_rows, raw_rows, exact_rows, frozen_rows = map(a.csv_rows, (a.ANCHOR_TEST, raw_csv, exact_csv, a.BASE_CSV))
    test_prior, _ = fit_prior(train, frames, range(len(train)))
    a.need(prior_agrees(test_prior, deployed.prior), "Shipped all-train prior mismatch")
    a.need(dict(deployed.rows("rescue-small", raw_rows.items())) == old_rows, "Exact historical test rescue replay failed")
    test_base, test_frozen, test_raw, test_old = ({i: m._parse(t, True, i) for i, t in rs.items()}
                                                for rs in (exact_rows, frozen_rows, raw_rows, old_rows))
    pp, ee = gated_replay(test_base, test_frozen, test_raw, test_old, deployed)
    test = dict(historic_exact_replay=True,
                historic_anchor_flips=sum(p["take"] != q["take"] for i in test_raw
                                          for p, q in zip(test_raw[i], test_old[i], strict=True)),
                events=event_summary(ee), export=None)
    if rep["robust"] and test["events"]["rows"]:
        text = {i: t if not ee[i] else ";".join(
            p if test_base[i][j]["take"] == pp[i][j]["take"] else p.rsplit(maxsplit=1)[0] + f" {pp[i][j]['take']}"
            for j, p in enumerate(t.split(";"))) for i, t in exact_rows.items()}
        path = OUT / "submission_reapply_rescue.csv"
        a.token_changes(exact_rows, text, {9})
        a.write_rows(path, text)
        a.need(a.csv_rows(path) == text, "Test export round trip")
        a.validate_csv(path)
        test["export"] = dict(path=str(path.relative_to(a.ROOT)), sha256=a.sha(path))
    rep["test"] = test
    rep["inputs_sha256"] = {str(p.relative_to(a.ROOT)): a.sha(p) for p in
                            (base_path, raw_path, fit_report_path, inner_path, cache_path, cache_report_path,
                             raw_csv, exact_csv, a.ANCHOR_TEST, a.BASE_CSV, a.BASE_OOF, DEFAULT_MODEL, a.MANIFEST)}
    a.dump(OUT / "reapply_report.json", rep)
    print("Exact test replay:", test["historic_anchor_flips"], "historical flips;", test["events"]["rows"],
          "changes on gated test;", "export", test["export"], flush=True)


if __name__ == "__main__":
    main()
