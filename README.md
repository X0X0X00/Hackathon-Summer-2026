# WYH Modeling Track

## Released models

| Version | Method | Validation | OOF | Official score |
| --- | --- | --- | ---: | --- |
| MODEL V1 | Hierarchical Signature Specialists | frozen 3-fold | 75.98% | Not submitted |
| MODEL V2 | Reference-only LightGBM + approved Zenodo reference | team-compatible 5-fold | 82.12% | Not submitted |

## MODEL V2

MODEL V2 is the current WYH candidate: a reference-only LightGBM trained on **136,574** cleaned cells from the approved Zenodo MERFISH spinal-cord deposit (record 18039571). Local 5-fold OOF accuracy is **82.12%** (4106 / 5000), **+6.14 percentage points** versus MODEL V1.

These figures are **WYH local validation results**. MODEL V1 and MODEL V2 were **not official captain submissions**. No official leaderboard score is claimed.

- Candidate: [`outputs/submissions/model_v2_candidate.csv`](outputs/submissions/model_v2_candidate.csv)
- MODEL V2 write-up: [`docs/versions/model_v2.md`](docs/versions/model_v2.md)
- MODEL V1 write-up: [`docs/versions/model_v1.md`](docs/versions/model_v1.md)

A predeclared V2-C blend (C1) reached 82.24% OOF and was rejected: the gain was only +6 net cells, folds 3–4 regressed, and macro-F1 decreased. That result is evidence of conservative model selection / overfitting control; it is not the released MODEL V2 score.

---

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
  
