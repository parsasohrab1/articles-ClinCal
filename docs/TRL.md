# TRL assessment – ClinCal

**Claimed level: TRL 4** (technology validated in the laboratory on public real data). Not TRL 5: no validation in a relevant clinical environment.

Data: UCI Heart Disease, four sites (Cleveland n=303; Hungarian, Switzerland, VA as targets). Source model: logistic regression trained on Cleveland (5-fold CV on source: AUC 0.859, ECE 0.040, slope 0.85).

## Model applied unchanged to other sites

| Site | n | Prevalence | Mean prediction | AUC | Brier | ECE | Slope | Intercept |
|---|---|---|---|---|---|---|---|---|
| hungarian | 294 | 0.36 | 0.41 | 0.880 | 0.131 | 0.064 | 0.89 | -0.42 |
| switzerland | 123 | 0.93 | 0.69 | 0.723 | 0.167 | 0.248 | 0.47 | 2.41 |
| va | 200 | 0.74 | 0.70 | 0.732 | 0.173 | 0.093 | 0.56 | 0.53 |

## Recalibration with k labelled target patients (mean over 200 splits; Brier / ECE on the remaining patients)

| Site | k | None | Intercept only | Platt | AUC Platt vs none |
|---|---|---|---|---|---|
| hungarian | 20 | 0.130 / 0.063 | 0.136 / 0.075 | 0.144 / 0.099 | 0.880 vs 0.880 |
| hungarian | 50 | 0.131 / 0.064 | 0.130 / 0.055 | 0.133 / 0.070 | 0.880 vs 0.880 |
| hungarian | 100 | 0.131 / 0.066 | 0.128 / 0.050 | 0.129 / 0.059 | 0.880 vs 0.880 |
| switzerland | 20 | 0.167 / 0.252 | 0.069 / 0.065 | 0.074 / 0.073 | 0.628 vs 0.728 |
| switzerland | 50 | 0.166 / 0.249 | 0.065 / 0.058 | 0.062 / 0.058 | 0.685 vs 0.725 |
| switzerland | 100 | not evaluable (fewer than 30 remaining test patients) | | | |
| va | 20 | 0.173 / 0.094 | 0.179 / 0.110 | 0.184 / 0.113 | 0.696 vs 0.733 |
| va | 50 | 0.174 / 0.097 | 0.174 / 0.100 | 0.172 / 0.082 | 0.730 vs 0.730 |
| va | 100 | 0.174 / 0.102 | 0.173 / 0.097 | 0.169 / 0.080 | 0.732 vs 0.732 |

## Findings and caveats
- Discrimination holds up (AUC 0.72 to 0.88) but calibration does not transfer: Switzerland (prevalence 0.94) has ECE 0.25 and intercept 2.4.
- With a large prevalence shift, intercept-only recalibration with only 20 labelled patients cuts Brier from 0.167 to about 0.069.
- With a small shift (Hungarian) recalibration with 20 patients makes things slightly worse (Brier 0.130 to 0.136; Platt 0.144) and only helps from about 100 patients (0.128).
- Platt scaling on few patients can reduce AUC (Switzerland 0.728 to 0.628 at k = 20) – a ranking change caused by a noisy slope estimate.
- VA: little gain from any recalibration at these sizes.
- Sites differ in protocol, selection and missing data (Switzerland and part of the other sites lack cholesterol or other fields, imputed with the source median); the shift is not purely a prevalence shift. The numbers describe these four public datasets only.
- Switzerland has 123 patients, so k = 100 leaves too few test patients and is not evaluated.
