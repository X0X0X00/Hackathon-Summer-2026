# University of Rochester Biomedical Data Science Hackathon Summer 2026
Welcome to the landing page for the hackathon. The hackathon will commence 8/18. It will be a prediction challenge. All predictions should be submitted through GitHub using the captain's handle. Scoring will also happen in GitHub. All details regarding the hackathon will be posted here.  

 Register for the hackathon [here](https://forms.gle/TEW1BHqezsKgTTKL9). Please make sure each individual competing on your team is fully registered. Each team needs a captain with a github handle. To receive a prize, you must supply your University of Rochester e-mail address. All teams scoring better than random will receive a participation prize. 1st and 2nd place winning teams in each division will get a cash prize (see below).
 **All team members must submit their own registration form to participate.**  

# Overview
This is a prediction challenge with spatial transcriptomics data. The objective of the hackathon is to correctly predict cell type labels in MERFISH_cell_type_annotation. Group performance will be measured by the confusion matrix overall accuracy: number of correct predictions / total number of predictions.

# Challenge description
The challenge is to classify cell types in a mouse neuronal tissue dataset collected using MERFISH, an imaging-based spatial transcriptomics technique that measures gene expression while preserving each cell's exact location in the tissue. The dataset covers ~10,000 cells and 200 genes, with sparse transcript counts paired with spatial coordinates and cell metadata (like cell volume, region, gender, and mouse ID). Participants must predict the cell type label for each cell in the test set. A full description of the challenge and dataset is here: [Data.Description.md](Data.Description.md).

# Logistics

0.   Each team must have a github handle associated with it in order to participate.  Make sure you edit your registration or email the organizers to provide this, if you haven't yet. Your team will not be scored if you do not provide a handle.
1.   You may add team members up
to noon EDT on 8/18 by editing your response to the google form or emailing the organizers.
2.  Teams of entirely undergraduates will be in the undergraduate
division, else they will be in the open division.
3. Further instructions for submitting predictions will be posted here as they become available
4.  Competition runs through 2:59 PM EDT 22-August-2026.  The predictions each team has committed to their repository at that time will be used to determine their final score. Captains must submit their own predictions. Any use of predictions from other teams is disqualifying. Winning teams must submit their code to organizers to claim their prize.

# Prizes
   
1.  First place in each division: $300 + $75 x (team size)
2.  Second place in each division: 0 + $50 x (team size)
  

## Team yhh — V7 submission

The current team submission is [`prediction/prediction.csv`](prediction/prediction.csv).

### Validation result

| Model | 5-fold OOF accuracy | Correct cells | Held-out folds 3–4 |
|---|---:|---:|---:|
| Reproduced V6 ensemble | 82.50% | 4,125 / 5,000 | 81.85% |
| **V7 hierarchical specialists** | **83.20%** | **4,160 / 5,000** | **82.30%** |

These are local out-of-fold validation results, not an official leaderboard score. V7 improves the reproduced V6 baseline by 35 correctly classified cells (+0.70 percentage points). Model selection used folds 0–2; folds 3–4 were retained as a holdout check.

### V7 method

V7 starts from the five-member V6 probability ensemble (`refonly_full`, `ext_all25`, `mlp3`, `yhh_v1`, and `bag8mix`), then applies the existing E/I and Segment/Laminae constraints. It adds gated hierarchical specialists only for ambiguous cells:

- oligodendrocyte-lineage family specialist;
- astrocyte/vascular/meningeal family specialist;
- targeted pair experts for `oligodendrocyte_1` vs `oligodendrocyte_progenitor_2`, `astrocyte_1` vs `endothelial`, and `meninges_1` vs `meninges_2`;
- external reference cells are used only as labeled reference data, while competition OOF predictions remain fold-aware.

Probability-only and anatomy-aware global routers, plus three rare neuronal pair experts, were evaluated but excluded because they did not improve the held-out folds. The accepted corrections are therefore limited to the specialist stages that improved both the tuning folds and the holdout check.

The optimization entry point is [`work/optimize_v7.py`](work/optimize_v7.py), and the complete selection report is [`work/v7_optimization_report.json`](work/v7_optimization_report.json). After preparing the V6 caches and member probability files described in [`work/README.md`](work/README.md), run:

```powershell
python work/optimize_v7.py
```
