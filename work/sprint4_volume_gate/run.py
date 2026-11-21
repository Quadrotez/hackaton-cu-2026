"""One predeclared local-plane bulk volume gate on the winning score-only rerank."""
from collections import Counter, defaultdict
import math
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent / "sprint4_form_rank"))
import run as anchor
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from threadpoolctl import threadpool_limits

CONFIG = dict(depth_min=.12, depth_max=3., in_mask_coverage=.80, floor_relative_p75=.02,
              ring_inner_px=4, ring_outer_px=20, ring_min_samples=96, floor_cosine_min=.9,
              floor_offset_ratio_min=.5, floor_offset_ratio_max=1.5,
              grid_m=.004, log_blend=.25, cap_factor=1.35, fit="real GT bulk only; mean log-ratio per object, then mean")
DEPTH_MIN, DEPTH_MAX = CONFIG["depth_min"], CONFIG["depth_max"]
RING_IN, RING_OUT = (cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*k+1, 2*k+1))
                     for k in (CONFIG["ring_inner_px"], CONFIG["ring_outer_px"]))
GATED = "gated"


def result(reason, coverage=0., error=None, ml=None, grid_cells=0, imputed_pixels=0):
    return dict(reason=reason, coverage=float(coverage), floor_error_p75=error, ml=ml,
                grid_cells=grid_cells, imputed_pixels=imputed_pixels)


def valid_depth(z):
    return (z >= DEPTH_MIN) & (z <= DEPTH_MAX)


def fit_inverse(x, y, z):
    # Inverse depth is affine in normalized camera ray coordinates. Weighted
    # least squares with weight z^2 fits errors relative to fitted depth.
    A = np.column_stack((x, y, np.ones(len(z))))
    az = A * z[:, None]
    beta, _, rank, _ = np.linalg.lstsq(az, np.ones(len(z)), rcond=None)
    if rank != 3:
        return None
    for _ in range(8):
        r = az @ beta - 1.
        sig = max(.002, 1.4826 * np.median(np.abs(r - np.median(r))))
        w = np.sqrt(np.minimum(1., 1.345 * sig / np.maximum(np.abs(r), 1e-12)))
        beta, _, rank, _ = np.linalg.lstsq(az * w[:, None], w, rcond=None)
        if rank != 3:
            return None
    return beta


def proxy(depth, m, offset, camera, nominal):
    """Metric integral of height over the orthogonal plane projection of the silhouette."""
    if not m.any():
        return result("empty_mask")
    z = depth.astype(float) * .001
    good = valid_depth(z)
    seen = good & m
    n_seen, n_mask = int(seen.sum()), int(m.sum())
    coverage = n_seen / n_mask
    if coverage < CONFIG["in_mask_coverage"] or n_seen < 1:
        return result("inside_coverage", coverage)
    m8 = m.astype("uint8")
    ring = (cv2.dilate(m8, RING_OUT) != 0) & (cv2.dilate(m8, RING_IN) == 0) & good
    v, u = np.nonzero(ring)
    if len(v) < CONFIG["ring_min_samples"]:
        return result("ring_support", coverage)
    off_x, off_y = offset
    fx, fy, cx, cy = (camera[k] for k in ("fx", "fy", "cx", "cy"))
    x, y, zz = (u + off_x - cx) / fx, (v + off_y - cy) / fy, z[v, u]
    # Fixed evenly-spaced subsampling only for the fit; the error uses all ring points.
    fit = np.linspace(0, len(v) - 1, min(len(v), 4096), dtype=int)
    beta = fit_inverse(x[fit], y[fit], zz[fit])
    if beta is None or not np.isfinite(beta).all() or np.linalg.norm(beta) == 0:
        return result("invalid_plane", coverage)
    inv = beta[0]*x + beta[1]*y + beta[2]
    error = float(np.quantile(np.abs(inv * zz - 1), .75))
    if error > CONFIG["floor_relative_p75"]:
        return result("ring_residual", coverage, error)
    # Plane equation n·X = d, with n from camera toward the fitted floor.
    # Metadata plane is only a validity check, never the integrated plane.
    n = beta / np.linalg.norm(beta)
    d = 1 / np.linalg.norm(beta)
    nominal_n = -np.asarray(nominal[:3])
    nominal_d = nominal[3] / np.linalg.norm(nominal_n)
    if (not np.all(inv > 0) or n @ (nominal_n / np.linalg.norm(nominal_n)) < CONFIG["floor_cosine_min"]
            or not CONFIG["floor_offset_ratio_min"] * nominal_d <= d <=
            CONFIG["floor_offset_ratio_max"] * nominal_d):
        return result("invalid_plane", coverage, error)
    v, u = np.nonzero(m)
    x, y = (u + off_x - cx) / fx, (v + off_y - cy) / fy
    inv = beta[0]*x + beta[1]*y + beta[2]
    if not np.all(inv >= 1 / DEPTH_MAX) or not np.all(inv <= 1 / DEPTH_MIN):
        return result("invalid_floor_depth", coverage, error)

    # For the <=20% missing mask pixels, impute the nearest measured *height*,
    # then reconstruct depth along that pixel's own ray. No missing sample
    # becomes a zero-height observation; a failed reconstruction rejects the gate.
    _, ids = cv2.connectedComponents(m8)
    filled = z.copy()
    for k in np.unique(ids[m & ~seen]):
        comp = ids == k
        c_good = comp & seen
        if not np.any(c_good):
            return result("component_no_depth", coverage, error)
        l, t, w, h = cv2.boundingRect(comp.astype("uint8"))
        win = np.s_[t:t+h, l:l+w]
        ii = distance_transform_edt(~c_good[win], return_distances=False, return_indices=True)
        miss_v, miss_u = np.nonzero((comp & ~seen)[win])
        src_v, src_u = ii[:, miss_v, miss_u]
        miss_v, miss_u, src_v, src_u = miss_v+t, miss_u+l, src_v+t, src_u+l
        def ray_dot(u, v):
            return n[0]*(u + off_x - cx)/fx + n[1]*(v + off_y - cy)/fy + n[2]
        # Equal distance from the fitted plane: n·X is held constant.
        filled[miss_v, miss_u] = z[src_v, src_u] * ray_dot(src_u, src_v) / ray_dot(miss_u, miss_v)
    zz = filled[v, u]
    if not np.all(valid_depth(zz)):
        return result("imputed_depth_range", coverage, error)
    xyz = np.column_stack((zz*x, zz*y, zz))
    height = np.maximum(0., d - xyz @ n)
    e1 = np.cross(n, np.eye(3)[np.argmin(np.abs(n))])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    # e1,e2 already lie in the floor plane, so dot products are the exact
    # coordinates after orthogonal projection. Integrate the top envelope.
    cell = CONFIG["grid_m"]
    cu = np.floor(xyz @ e1 / cell).astype(np.int64)
    cv = np.floor(xyz @ e2 / cell).astype(np.int64)
    key = (cu - cu.min()) * (int(cv.max() - cv.min()) + 1) + (cv - cv.min())
    order = np.argsort(key)
    starts = np.r_[0, np.flatnonzero(np.diff(key[order])) + 1]
    tops = np.maximum.reduceat(height[order], starts)
    ml = float(tops.sum() * cell**2 * 1e6)
    if not np.isfinite(ml) or ml <= 0:
        return result("nonpositive_integral", coverage, error)
    return result(GATED, coverage, error, ml, len(tops), n_mask - n_seen)


class Geometry:
    def __init__(self, camera):
        self.camera = camera
        self.input_hashes = {}
        self.domains = defaultdict(set)
        self.prev = None

    def frame(self, rgb, shape, domain):
        paths = tuple(rgb.with_name(n) for n in ("depth.png", "meta.json"))
        if self.prev != paths:
            self.depth = cv2.imread(str(paths[0]), cv2.IMREAD_UNCHANGED)
            meta = anchor.read(paths[1])
            self.plane = tuple(meta["floor_plane"])
            for f in paths:
                self.input_hashes[str(f.relative_to(anchor.ROOT))] = anchor.sha(f)
            self.prev = paths
        anchor.need(self.depth is not None and self.depth.shape == shape and self.depth.dtype == np.uint16,
                    f"Bad depth: {paths[0]}")
        h, w = shape
        anchor.need((h,w) == (self.camera["height"], self.camera["width"]) and
                    np.isfinite(self.plane).all() and self.plane[-1] > 0 and np.linalg.norm(self.plane[:3]) > 0,
                    f"Invalid camera domain: {paths[0]}")
        self.domains[domain].add((h, w, *self.plane))

    def gt(self, rec):
        _, im, a, _, _ = rec
        h, w = im["height"], im["width"]
        self.frame(anchor.DATA / im["file_name"], (h, w), "train_gt")
        mask = anchor.polygon_mask(a["segmentation"], w, h, 0, 0)
        return self.measure(mask)

    def predicted(self, src, domain):
        _, _, rgb, _, packed, (h,w) = src
        self.frame(rgb, (h, w), domain)
        mask = np.unpackbits(packed, axis=-1, count=w).astype(bool)
        return self.measure(mask)

    def measure(self, mask):
        x, y, w, h = cv2.boundingRect(mask.astype("uint8"))
        if not w or not h:
            return result("empty_mask")
        p = CONFIG["ring_outer_px"] + 1
        t, l, b, r = max(y-p,0), max(x-p,0), min(y+h+p,mask.shape[0]), min(x+w+p,mask.shape[1])
        return proxy(self.depth[t:b,l:r], mask[t:b,l:r], (l,t), self.camera, self.plane)


def fit_scale(train, gtrain, ix):
    by_object = defaultdict(list)
    eligible = 0
    for j in ix:
        if j not in gtrain:
            continue
        eligible += 1
        ml = gtrain[j]["ml"]
        _, _, a, obj, _ = train[j]
        if ml is not None:
            target = a["attributes"]["volume_ml"]
            anchor.need(target > 0, f"Invalid GT volume {obj}")
            by_object[obj].append(math.log(target / ml))
    anchor.need(by_object, "Cannot fit a log scale without valid bulk instances")
    logscale = float(np.mean([np.mean(by_object[o]) for o in sorted(by_object)]))
    return dict(log_scale=logscale, scale=math.exp(logscale), gt_bulk_rows=eligible,
                gated_gt_rows=sum(map(len, by_object.values())), gated_gt_objects=len(by_object))


def update(pred, alternate_slots, geom, logscale, sources):
    out = []
    for p, j in zip(pred, alternate_slots, strict=True):
        if sources[j][3]["form"] == "bulk" and geom[j]["ml"] is not None and p["volume"] > 0:
            base = p["volume"]
            d = CONFIG["log_blend"] * (math.log(geom[j]["ml"] / base) + logscale)
            cap = math.log(CONFIG["cap_factor"])
            p = dict(p, volume=base * math.exp(max(-cap, min(cap, d))))
        out.append(p)
    return out


def coverage(mapping, base, sources, geom):
    eligible = [j for i in base for j in mapping[i] if sources[j][3]["form"] == "bulk"]
    ctr = Counter(geom[j]["reason"] for j in eligible)
    unique = set(eligible)
    return {"eligible_rows": len(eligible), "gated_rows": ctr[GATED],
            "gated_frames": sum(any(j in geom and geom[j]["reason"] == GATED for j in js) for js in mapping.values()),
            "reasons": dict(ctr), "unique_bulk_donors": len(unique),
            "unique_gated_donors": sum(geom[j]["reason"] == GATED for j in unique)}


def changed(gt, old, new):
    rows, frames, hits, hit_frames = 0, 0, 0, 0
    for i in gt:
        a, b = old[i], new[i]
        need = anchor.need
        for p, q in zip(a, b, strict=True):
            need({k:v for k,v in p.items() if k != "volume"} == {k:v for k,v in q.items() if k != "volume"},
                 f"Non-volume change {i}")
        n = sum(p["volume"] != q["volume"] for p, q in zip(a, b, strict=True))
        rows += n
        frames += n > 0
        ia, ib = ({id(p): j for j,p in enumerate(ps)} for ps in (a,b))
        changed_matches = 0
        for p,q in zip(anchor.metric._match(gt[i], a), anchor.metric._match(gt[i], b), strict=True):
            need(ia.get(id(p)) == ib.get(id(q)), f"Match slot changed {i}")
            changed_matches += p is not None and p["volume"] != q["volume"]
        hits += changed_matches
        hit_frames += changed_matches > 0
    return dict(predictions=rows, frames=frames, matched_gts=hits, matched_frames=hit_frames)


def audit_mapping(base, mapping, sources):
    for i, ps in base.items():
        by_raw_slot = defaultdict(list)
        for p, slot in zip(ps, mapping[i], strict=True):
            anchor.need(i == sources[slot][0], f"Cross-frame donor {i}")
            by_raw_slot[slot].append(p)
        for slot, members in by_raw_slot.items():
            form = sources[slot][3]["form"]
            parent = [p for p in members if p["form"] == form]
            # If alternate is present its base volume must be that of the matching donor.
            for p in members:
                if p["form"] != form:
                    anchor.need(any(p["volume"] == q["volume"] for q in parent), f"Incorrect volume donor {i}")


def main():
    camera_p = anchor.DATA / "camera.json"
    camera = anchor.read(camera_p)
    anchor.need(camera["depth_unit"] == "mm" and camera["depth_aligned_to"] == "color", "Bad camera depth domain")
    specs = anchor.read(anchor.MANIFEST)["folds"]
    frames, gt, objects = anchor.labelled()
    hashes = {}
    frozen, omap, vs, _, _, tmap, ts = anchor.prepare_sources(frames, specs, hashes)
    omap = {i: keys for i, (_, keys) in omap.items()}
    fold_path = anchor.OUT / "report.json"
    rank_report = anchor.read(fold_path)
    baseline_oof = anchor.OUT / "selected_oof.json"
    baseline_csv = anchor.OUT / "submission_selected.csv"
    for p, key in ((baseline_oof, "oof"), (baseline_csv, "csv")):
        anchor.need(anchor.sha(p) == rank_report["scores"][key], f"Rank input SHA changed: {p}")
    old = anchor.read(baseline_oof)
    rows = anchor.csv_rows(baseline_csv)
    test = {i: anchor.metric._parse(s, True, i) for i,s in rows.items()}
    anchor.churn(gt, frozen, old)
    audit_mapping(old, omap, vs)
    audit_mapping(test, tmap, ts)
    geo = Geometry(camera)
    train = anchor.training_records(frames)
    gtrain = {j: geo.gt(r) for j,r in enumerate(train) if r[4] == "bulk" and frames[r[0]][0] == "real"}
    vg = {j: geo.predicted(src, "real_oof") for j,src in enumerate(vs) if src[3]["form"] == "bulk"}
    tg = {j: geo.predicted(src, "real_test") for j,src in enumerate(ts) if src[3]["form"] == "bulk"}
    domain_match = geo.domains["real_test"] <= geo.domains["real_oof"] and bool(geo.domains["real_test"])
    print("Metric geometry complete:", len(gtrain), len(vg), len(tg), "test domain covered:", domain_match, flush=True)
    pred = {}
    scales, folds = {}, {}
    for spec in specs:
        k = spec["fold"]
        ix = anchor.fit_rows(train, spec)
        scales[k] = fit_scale(train, gtrain, ix)
        fg = {i: gt[i] for i in spec["val_real_image_ids"]}
        for i in fg:
            pred[i] = update(old[i], omap[i], vg, scales[k]["log_scale"], vs)
        folds[k] = dict(anchor.paired(fg, old, pred), changed=changed(fg, old, pred),
                        gate=coverage({i: omap[i] for i in fg}, fg, vs, vg))
    pooled = anchor.paired(gt, old, pred)
    loo = anchor.removal(gt, objects, old, pred)
    invariant = all(d["delta"][m] == 0 for d in (pooled, *folds.values()) for m in ("D", "M", "S"))
    anchor.need(invariant, "Unexpected D/M/S change")
    passed = (pooled["delta"]["Score"] > 0 and sum(x["delta"]["Score"] > 0 for x in folds.values()) >= 3
              and loo["min_delta"] > 0 and domain_match)

    scale_test = fit_scale(train, gtrain, range(len(train)))
    output = OUT / "submission_best_verified.csv"
    if passed:
        to_export = {i: anchor.replace_tokens(rows[i], update(test[i], tmap[i], tg, scale_test["log_scale"], ts),
                                               {7: "volume"}) for i in rows}
        anchor.write_rows(output, to_export)
    else:
        output.write_bytes(baseline_csv.read_bytes())
        anchor.need(anchor.sha(output) == anchor.sha(baseline_csv), "Fallback was not byte-identical")
    validated = anchor.csv_rows(output)
    from_rank = anchor.token_changes(rows, validated, {7} if passed else set())
    from_frozen = anchor.token_changes(anchor.csv_rows(anchor.BASE_CSV), validated, {4, 7} if passed else {4})
    anchor.need(anchor.test_counts(validated) == (137, 342, 10), "Incorrect final inventory")
    anchor.validate_csv(output)
    hashes.update({p: anchor.sha(p) for p in (camera_p, fold_path, baseline_oof, baseline_csv, anchor.BASE_CSV,
        anchor.BASE_OOF, anchor.MANIFEST, Path(__file__).resolve(), Path(anchor.__file__).resolve(),
        anchor.WBF / "base_detections.jsonl", anchor.WBF / "base_masks.npz",
        anchor.DATA / "labels/real_train.json", anchor.DATA / "labels/synth_train.json")})
    anchor.dump(OUT / "report.json", dict(config=CONFIG, selected="gated" if passed else "rank_fallback",
        rank_variant=rank_report["selected"], pooled=pooled, folds=folds, best_object_removal=loo,
        changed=changed(gt, old, pred), gate=coverage(omap, old, vs, vg), test_gate=coverage(tmap, test, ts, tg),
        train_scales=scales, test_scale=scale_test, domain_verified=domain_match,
        geometry_domains={k: sorted(v) for k,v in geo.domains.items()},
        output_sha256=anchor.sha(output), diff_vs_rank=from_rank, diff_vs_original_frozen=from_frozen,
        test_counts=list(anchor.test_counts(validated)),
        hashes={str(p.relative_to(anchor.ROOT)):h for p,h in hashes.items()}, depth_and_meta_sha256=geo.input_hashes))
    print("Volume", pooled["delta"], "folds", [round(x["delta"]["Score"], 5) for x in folds.values()],
          "min LOO", loo["min_delta"], "passed", passed, anchor.sha(output))


if __name__ == "__main__":
    cv2.setNumThreads(1)
    with threadpool_limits(2):
        main()
