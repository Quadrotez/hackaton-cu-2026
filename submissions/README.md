# Astramis Case Submissions Ledger

This directory contains the verified top-performing submissions for the Astramis hackathon case (`stackmorelayers.ai/competitions/astramis`), along with their public leaderboard scores, SHA-256 hashes, and architectural lineage.

---

## 🏆 Verified Leaderboard Ledger

| Submission File | Public LB | Key Improvements & Architecture | SHA-256 Checksum |
| :--- | :---: | :--- | :--- |
| **`submission_83_91_champion.csv`** | **83.9142** | **All-time Champion**: DINO+CLIP multimodal consensus + empty border frame pruning + strict physical take consistency. | `df4155ff6a53e34e389d328ddae0f2d9862c9c635a0529085624e2fca777a98c` |
| **`submission_83_91_v17_overhaul.csv`** | **83.9142** | **Multimodal Overhaul**: 5-fold DINO ViT-B/16 + CLIP ViT-L/14 ensemble. 26 material tokens updated (17 false-positive metals reclassified to wood/plastic). | `9c1a6d346a063ebeb98c64535ba9b589f98b6114124e67239fefeafb33fc1d28` |
| **`submission_83_82_v11.csv`** | **83.8210** | **Clean Material Boost**: 21 high-margin DINO material replacements ($p \ge 0.70, \text{margin} \ge 0.20$) on top of v2. | `e6b18ee8a0d70be7ea8d5b732bfedbc90554d10fb5dedda52064df8aebc7c4fa` |
| **`submission_83_64_v2.csv`** | **83.6421** | **Physical Take Consistency**: Enforced physical bounds: $\text{mass} < 250\text{g} \to \text{take}=1$; $\text{mass} > 1000\text{g} \to \text{take}=0$. | `c632268bcbe098084993dd706c3f4773af4f985f23ad819efc2aa4bf1db8ee4b` |

---

## 🔬 Methodology & Core Insights

### 1. Metric Decomposition
Competition metric:
$$\text{Score} = 100 \times (0.40 \cdot D + 0.20 \cdot M + 0.25 \cdot V + 0.15 \cdot S)$$
- **$D$ (Form mAP50, bulk vs solid)**: Detector output ranking.
- **$M$ (Material Macro Recall)**: Balanced accuracy across 6 classes (`metal`, `wood`, `plastic`, `glass`, `paper`, `mineral`).
- **$V$ (Volume Accuracy)**: Solid volume ratio in YOLO was already 0.9971 (near optimal); unconditioned scaling degrades score.
- **$S$ (Take Decision)**: Binary decision for mass $\le 500\text{g}$.

### 2. Physical Consistency ($S$)
In baseline YOLO, heavy objects (>1000g) occasionally had `take=1` and very light objects (<250g) had `take=0` due to classifier noise.
Enforcing physical boundaries derived from $\text{mass} = \text{volume} \times \text{density}$ yielded an immediate jump from **82.0385 $\to$ 83.6421** (+1.60 pts).

### 3. Multimodal Material Ensemble ($M$)
YOLO baseline detector had a severe false-positive bias towards `metal` (Macro Recall 66.19% on OOF).
A 5-fold ensemble combining **DINO ViT-B/16** (geometric texture representation) and **CLIP ViT-L/14** (semantic representation) lifted OOF Macro Recall to **78.53%** (+12.34% absolute gain).
Applying high-confidence overrides yielded **83.9142** on the public leaderboard.

---

## 🛠 Reproduction & Validation

To regenerate the multimodal material overhaul:
```bash
python fusion/overhaul_materials.py \
  --base submissions/submission_83_91_champion.csv \
  --out submissions/submission_83_91_v17_overhaul.csv \
  --test-probs fusion/multimodal_probs/multimodal_test_probs.csv
```

To validate format and structure against the competition test set:
```bash
python -c "
import sys
sys.path.append('fusion')
import submission_tools
from pathlib import Path

for path in Path('submissions').glob('*.csv'):
    submission_tools.validate_csv(path)
"
```
All files strictly contain 137 test frames (1 header + 137 rows), semicolons separating prediction boxes, and space-separated attribute tokens.
