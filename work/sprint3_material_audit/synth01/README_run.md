# synth01: protocol for the certified subset

Frozen `bulk_y_bottom` is the **same** anchor as in `run01`. Its provenance is not newly certified: the background check applies only to synth admitted by this experiment. The matched comparison is `certified synth – real-only interior`, not `certified synth – run01` (the descriptors now use eroded ROIs in both arms).

Before feature extraction, admit only frames on each fold's supplied background whitelist; independently check consistency with the detector manifest and the mapped empty-real donor not being held out. Require one annotation and one mask instance, a train object of the pair, and real train views of that parent. Certify one and only one train real source/translation/optional horizontal reflection: compare all pixels in the 4px-eroded synth support, requiring source pixels to be inside the 2px-eroded real annotation, identical depth validity plus a global depth offset up to 100mm with at most 1mm integer quantization spread, and channelwise monotone RGB lookup mappings across the entire support (colors need not be identical). Require at least 512 support pixels / 256 valid depth pixels, ≥16 distinct original values per color channel and in valid depth. Horizontal translation comes from bbox origins ±2 pixels (accounting for reflection), with dimensions within 4px; vertical shift is zero. No approximated/partial certificates, no val-positive search. These fixed screens certify pixel-level correspondence on the inspected ROI, not a generator manifest or unseen pixels.

For both arms, descriptors use the same `run01` implementation on a 5px-eroded mask with all input outside a 4px-eroded support zeroed. Thus 3×3 Sobel and median neighborhoods for every scored pixel are contained inside certified support, and boundary features use the inner mask; box area/position/IDs are not features. An unscoreable OOF detection remains frozen in both arms. Same fixed L2 `C`, thresholds, train-object jackknife, class and parent totals as `run01`. Synth treatment redistributes relative weight `80:20` across each eligible parent's real/synth views (uniform separately in each domain); parents with no certified synth use `100:0`. The real-only arm always uses `100:0`. No metric-driven selection.

```sh
set -o pipefail
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B \
work/sprint3_material_audit/synth01/run_certified.py \
  2>&1 | tee work/sprint3_material_audit/synth01/certified.log
```

The default output directory is `certified/`; pass `--out work/sprint3_material_audit/synth01/certified_replica` for a repeat. It contains `report.json` (per-fold and pooled components, confusion and directional churn), `certificates.json` (synth source, offset, flip, verified count, parent and background), and both OOF prediction files.

## Results

| Fold | Allowed after background filter | Examined foreground | Certified synth views / parents | Real views / parents |
|---|---:|---:|---:|---:|
| 1 | 299 | 199 | 39 / 12 | 93 / 17 |
| 2 | 471 | 210 | 49 / 15 | 88 / 17 |
| 3 | 462 | 216 | 43 / 13 | 88 / 18 |
| 4 | 174 | 16 | 0 / 0 | 98 / 18 |
| 5 | 466 | 209 | 50 / 15 | 97 / 18 |

181 fold-specific synth observations, 55 unique scenes across folds, including 19 reflected scenes. Fold 4 is a no-treatment fold (`100:0` for every parent); the zero delta there says nothing about synth. The result is limited to the certified subset and its fixed weights, not the entire synth dataset.

| Contrast | Pooled Δ | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---|---:|---:|---:|---:|---:|---:|
| Synth – real-only interior | 0 | 0 | 0 | 0 | 0 | 0 |
| Real-only interior – frozen | −0.1732 | 0 | 0 | 0 | −0.7792 | −0.2564 |
| Synth – frozen | −0.1732 | 0 | 0 | 0 | −0.7792 | −0.2564 |

Both new OOF files are byte-identical (Score **72.6070**, `D/V/S` unchanged). Synth vs matched control: **0 changed / 0 fixed / 0 broken**. Each vs frozen: **6 changed / 0 fixed / 3 broken**, plus 3 of the 6 changes unmatched; 5 switches `plastic→metal`, 1 `metal→plastic`. No test export, further search or change to `run01`.
