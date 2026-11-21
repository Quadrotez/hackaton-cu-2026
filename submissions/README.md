# Astramis Case Submissions Ledger

This directory contains the verified top-performing submissions for the Astramis hackathon case (`stackmorelayers.ai/competitions/astramis`), along with their public leaderboard scores, SHA-256 hashes, and architectural lineage.

---

## 🏆 Verified Leaderboard & Candidate Ledger

| Submission File | Public LB / Status | Key Improvements & Architecture | SHA-256 Checksum |
| :--- | :---: | :--- | :--- |
| **`submission_v18_joint_champion.csv`** | **Target Candidate** | **Joint Multi-Front Champion**: Merges Form Score Modulation ($D$ +0.1078 OOF) + Multimodal Material Overhaul ($M$ 66.19% $\to$ 78.53%) + Physical Take Consistency ($S$). | `9f69b3ac3c0962d406e406433c3155209174c4e18f9981e2fe915ae5617f7a08` |
| **`submission_v16_form_boost.csv`** | **Candidate** | **Form Score Calibration ($D$)**: Smooth logit score modulation ($\alpha=0.35$) on champion v13 using DINO+CLIP $P(\text{bulk})$ and $P(\text{solid})$. | `78624b384eeb01377ea6ba0b9a1414d1efb279a337bdd86b74ce75c2c09ba220` |
| **`submission_83_91_v17_overhaul.csv`** | **83.9142** | **Multimodal Overhaul ($M$)**: 5-fold DINO ViT-B/16 + CLIP ViT-L/14 ensemble. 26 material tokens updated (17 false-positive metals reclassified to wood/plastic). | `9c1a6d346a063ebeb98c64535ba9b589f98b6114124e67239fefeafb33fc1d28` |
| **`submission_83_91_champion.csv`** | **83.9142** | **All-time Champion**: DINO+CLIP multimodal consensus + empty border frame pruning + strict physical take consistency. | `df4155ff6a53e34e389d328ddae0f2d9862c9c635a0529085624e2fca777a98c` |
| **`submission_83_82_v11.csv`** | **83.8210** | **Clean Material Boost**: 21 high-margin DINO material replacements ($p \ge 0.70, \text{margin} \ge 0.20$) on top of v2. | `e6b18ee8a0d70be7ea8d5b732bfedbc90554d10fb5dedda52064df8aebc7c4fa` |
| **`submission_83_64_v2.csv`** | **83.6421** | **Physical Take Consistency**: Enforced physical bounds: $\text{mass} < 250\text{g} \to \text{take}=1$; $\text{mass} > 1000\text{g} \to \text{take}=0$. | `c632268bcbe098084993dd706c3f4773af4f985f23ad819efc2aa4bf1db8ee4b` |

---

## 🔬 Methodology & Core Insights

### 1. Metric Decomposition
Competition metric:
$$\text{Score} = 100 \times (0.40 \cdot D + 0.20 \cdot M + 0.25 \cdot V + 0.15 \cdot S)$$
- **$D$ (Form mAP50, bulk vs solid)**: Detector ranking. Logit score modulation with DINO+CLIP probabilities elevates pooled $D$ from **0.76695 $\to$ 0.87476** (+0.10781 across 5/5 folds).
- **$M$ (Material Macro Recall)**: Balanced accuracy across 6 classes. DINO+CLIP multimodal ensemble elevates OOF Macro Recall from **66.19% $\to$ 78.53%** (+12.34% absolute gain).
- **$V$ (Volume Accuracy)**: Solid volume ratio in YOLO was already 0.9971 (near optimal); unconditioned scaling degrades score.
- **$S$ (Take Decision)**: Binary decision for mass $\le 500\text{g}$.

### 2. Form Score Modulation Formula ($D$)
$$p = \begin{cases} p_{\text{bulk}} & \text{if } \text{form} = \text{'bulk'} \\ 1 - p_{\text{bulk}} & \text{if } \text{form} = \text{'solid'} \end{cases}$$
$$\text{logit}_{\text{old}} = \ln\left(\frac{s_{\text{old}}}{1 - s_{\text{old}}}\right)$$
$$\text{logit}_{\text{new}} = \text{logit}_{\text{old}} + \alpha \cdot (2p - 1)$$
$$s_{\text{new}} = \text{clip}\left(\sigma(\text{logit}_{\text{new}}), 0.0051, 0.9949\right)$$
Using $\alpha = 0.35$ avoids score collapse and preserves continuous VOC AP ranking without hard suppression.

### 3. Physical Consistency ($S$)
In baseline YOLO, heavy objects (>1000g) occasionally had `take=1` and very light objects (<250g) had `take=0` due to classifier noise.
Enforcing physical boundaries derived from $\text{mass} = \text{volume} \times \text{density}$ yielded an immediate jump from **82.0385 $\to$ 83.6421** (+1.60 pts).

---

## 🛠 Reproduction & Validation

To generate the joint multi-front champion submission (`v18`):
```bash
python fusion/optimize_form_scores.py \
  --sub-in submissions/submission_83_91_v17_overhaul.csv \
  --sub-out submissions/submission_v18_joint_champion.csv \
  --alpha 0.35
```

To validate format and structure against the competition test set:
```bash
python -c "
import sys
sys.path.append('fusion')
sys.path.append('work')
import submission_tools
from pathlib import Path

for path in Path('submissions').glob('*.csv'):
    submission_tools.validate_csv(path)
"
```
All files strictly contain 137 test frames (1 header + 137 rows), semicolons separating prediction boxes, and space-separated attribute tokens.
