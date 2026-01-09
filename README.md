# Time Series Multi-Teacher KD (Ensembles + Student Distillation)

This project:
- Evaluates **all teacher combinations** of: PatchTST, DLinear, PAttn, TimeMixer
- Forecast horizons: 1/2/3 months
- Datasets: Wfp_rice.csv and Wfp_wheat.csv (column: `price`)
- Builds an ensemble (mean) for each teacher combo
- Distills each ensemble into two students: **MLP** and **KAN**
- Logs MAE@1/2/3, AvgMAE, WorstMAE for val/test

## Setup (Colab)
```bash
pip install -r requirements.txt
```

## Run
```bash
python run.py
```

## Outputs
- `outputs/results.csv` contains metrics for:
  - ensemble (val/test)
  - student_mlp (val/test)
  - student_kan (val/test)

## Notes
- Teachers in this scaffold are lightweight placeholders; replace teacher files with full implementations later.
- Distillation assumes **shuffle=False** for the training loader to keep teacher targets aligned.
