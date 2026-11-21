# Metric bulk volume gate

From the repository root, after `work/sprint4_form_rank/run.py`:

```sh
PYTHONDONTWRITEBYTECODE=1 work/venv_yolo/bin/python work/sprint4_volume_gate/run.py
```

- Geometry uses the full verified predicted n640 mask on each real outer fold,
  and the verified WBF mask behind the same test anchor rule. All rows owned by
  a bulk anchor, including its alternate form, inherit the same geometry; no
  geometry is borrowed from a solid donor. The alternate's frozen volume must
  match its donor's before adjustment.
- Fit a Huber-IRLS affine inverse-depth plane on valid depth in a 4–20 px
  exterior mask ring. Require depth in `[0.12, 3]` m, ≥80% in-mask coverage,
  ≥96 ring samples, ring relative error p75 ≤2%, positive fitted range
  over the mask, and a plausible floor normal/offset compared with the
  capture calibration. Metadata is only a plausibility check, not the
  source of fitted heights.
- For missing pixels in a connected mask component, reconstruct depth from
  the nearest measured height in that component; reject unsupported components
  and invalid reconstructions. Back-project with camera intrinsics, project
  orthogonally onto the fitted floor and integrate positive top-envelope
  heights on a 4 mm grid in m³, converted to ml.
- Calibrate a single log-scale from eligible real GT bulk masks only (synthesis
  excluded for camera domain). Average log-ratios within each physical
  `object_id`, then across objects, restricting fit to the outer manifest train
  images/objects. Deployment fits on all labelled real bulk. Apply .25 of the
  calibrated log difference, clipped to ±log(1.35), or otherwise keep frozen
  volume. No settings are selected by outer results.
- Promote only when pooled gain and minimum gain after removing any real
  object are positive, at least three folds improve, D/M/S and matching
  are unchanged, and the test camera/depth/plane domain is covered by OOF.
  Otherwise the output is a byte-identical copy of the chosen form CSV.

`report.json` records all component scores, gates, fit counts, domain
signatures, token changes, leave-one-object-out gains and hashes. The only
final CSV is `submission_best_verified.csv`.
