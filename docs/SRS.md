# Software Requirements Specification – ClinCal

- SR-1 `data.load_site` reads the UCI processed files; `chol == 0` is treated as missing; target is `num > 0`; ten features common to all sites.
- SR-2 `calib.fit_model`: median imputation, standardisation, logistic regression (source site only).
- SR-3 `calib.metrics`: AUC, Brier score, equal-count-bin ECE (5 bins), calibration slope and intercept (logistic fit on logit of prediction), prevalence, mean prediction.
- SR-4 `recalibrate_intercept` (recalibration-in-the-large) and `recalibrate_platt` (slope + intercept) fitted on k labelled target patients.
- SR-5 `research/run_study.py`: 200 random splits per k in {20, 50, 100}; evaluation on the remaining target patients; repetitions with a one-class calibration sample or fewer than 30 test patients are skipped.
- SR-6 Four unit tests (ECE, calibration slope, intercept recalibration, Platt on shifted labels).
