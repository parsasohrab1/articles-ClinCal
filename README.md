# articles-ClinCal

Calibration of a clinical risk model under **cross-site shift**, on real patient data
(UCI Heart Disease: Cleveland, Hungarian, Switzerland, VA Long Beach). TRL 4 (laboratory validation on public real data; no prospective or clinical-setting validation).

```
pip install -e ".[dev]" && python -m pytest
# data: https://archive.ics.uci.edu/dataset/45/heart+disease (not committed)
python research/run_study.py /path/to/heart+disease
```
Outputs `research/results.json`. See `docs/` for BRS, SRS and TRL (with the full result tables).
