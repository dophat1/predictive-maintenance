# MetroPT-3 Predictive Maintenance: A to Z Guide

## Executive summary

This project asks whether sensor readings from one metro train compressor can provide an early warning before a documented air-leak failure.

The dataset contains 1,516,948 raw readings and four reported failure incidents. We summarize the readings into 10-minute windows, then ask whether a failure report begins during the next 24 hours. We compared logistic regression, random forest, and histogram gradient boosting using chronological event validation. Random forest ranked highest on the two earlier validation incidents, so it was selected for a final test on the later July period.

On that holdout, the model flagged 84 of 103 windows labeled as pre-failure (81.6% recall). It also flagged 4,943 windows labeled as not pre-failure. Only 84 of 5,027 flagged windows were positive (1.7% precision). Its PR-AUC was 0.012. The result is weak as an operational alarm: the model raises too many alerts and the single held-out incident is not enough to estimate future performance reliably.

**Business conclusion:** this is a research baseline, not a system to dispatch maintenance work orders. The useful next step is an engineer-reviewed, prospective shadow trial after grouping repeated alerts into episodes.

## 1. What is the dataset?

MetroPT-3 is a public dataset from the UCI Machine Learning Repository. It contains real operational measurements from an air production unit compressor on a metro train, collected over several months in 2020. The source has approximately 1.5 million records, sampled at about 1 Hz, and includes analog measurements and digital equipment-state signals.

The dataset includes four documented high-stress air-leak incidents:

| Incident | Reported start | Reported end |
|---|---|---|
| 1 | 2020-04-18 00:00 | 2020-04-18 23:59 |
| 2 | 2020-05-29 23:30 | 2020-05-30 06:00 |
| 3 | 2020-06-05 10:00 | 2020-06-07 14:30 |
| 4 | 2020-07-15 14:30 | 2020-07-15 19:00 |

Source: [UCI MetroPT-3 Dataset](https://archive.ics.uci.edu/dataset/791/metropt%2B3%2B), DOI [10.24432/C5VW3R](https://doi.org/10.24432/C5VW3R). The source dataset is licensed CC BY 4.0.

The data has one compressor and few failure events. It is not a representative fleet-wide sample, and the incident reports are not a complete maintenance history.

## 2. What does each original data field mean?

The UCI file has a timestamp, an index, seven analog sensor values, and eight digital signals. Analog pressure, temperature, and current fields are the model inputs in this project. Digital controls are described here for context but are not currently included in model features.

| Raw field | Meaning | Used by this model? |
|---|---|---|
| `index` | Row/index identifier in the source file. | No; it is not a physical measurement. |
| `timestamp` | Time at which the sensor record was logged. | Yes; used to order data, aggregate windows, label incidents, and split chronologically. |
| `TP2` | Pressure measured at the compressor, in bar. | Yes. |
| `TP3` | Pressure generated at the pneumatic panel, in bar. | Yes. |
| `H1` | Pressure associated with pressure drop when the cyclonic separator filter discharges, in bar. | Yes. |
| `DV_pressure` | Pressure drop associated with the air-dryer tower discharge, in bar. | Yes. |
| `Reservoirs` | Downstream reservoir pressure, in bar. | Yes. |
| `Oil_temperature` | Compressor oil temperature, in degrees Celsius. | Yes. |
| `Motor_current` | Current on one motor phase, in amperes. Its level varies with off/on/load/start states. | Yes. |
| `COMP` | Digital air-intake-valve state. | No in current baseline. |
| `DV electric` | Electrical control for the compressor outlet valve. | No in current baseline. |
| `TOWERS` | Which air-drying tower is operating. | No in current baseline. |
| `MPG` | Signal associated with starting the compressor under load when pressure falls. | No in current baseline. |
| `LPS` | Low-pressure switch signal. | No in current baseline. |
| `Pressure Switch` | Signal for air-drying tower discharge. | No in current baseline. |
| `Oil Level` | Signal indicating oil below expected level. | No in current baseline. |
| `Caudal Impulse` | Pulse count related to air flow from the unit to reservoirs. | No in current baseline. |

The UCI documentation is the authority for sensor definitions. Names such as `TP2` are source-system identifiers; they are not invented feature names.

## 3. What are we trying to predict?

The target is: **will a documented failure begin in the next 24 hours?**

The raw 1 Hz stream is aggregated into 10-minute bins. Each bin represents the recent state of the machine. A bin is labeled positive if its timestamp falls in the 24 hours immediately before one of the documented failure starts. Samples inside the reported active failure period are removed. Other usable bins are labeled negative.

This is a narrow, retrospective warning experiment. It does not predict the type or physical cause of failure. It does not estimate remaining useful life. “Negative” means “not inside a known 24-hour pre-failure window,” not a guarantee that the machine was perfectly healthy.

## 4. How are model features built?

There are seven analog inputs. For each one, the pipeline calculates:

1. **10-minute mean:** the average sensor value within each ten-minute period.
2. **10-minute standard deviation:** how much the sensor varied during that period.
3. **One-hour change:** current 10-minute mean minus the mean six bins earlier (six bins × ten minutes = one hour).

That produces 21 numeric features. Examples:

- `TP3__mean`: average pneumatic-panel pressure in that 10-minute window.
- `TP3__std`: within-window pressure variability.
- `TP3__mean__delta_1h`: change in average TP3 pressure over the preceding hour.
- `Motor_current__mean`: average motor current in that window.
- `Oil_temperature__mean__delta_1h`: one-hour change in average oil temperature.

Small gaps are interpolated for at most three bins; remaining incomplete rows are dropped. Sensor summaries are based on the current and preceding readings only. Future measurements are not used to create a feature.

## 5. Why use a machine-learning model?

A fixed rule such as “alert whenever pressure drops below X” may miss combinations of signals or normal operating differences. A model can learn relationships among multiple measurements and their recent variability/trend, then produce a risk score for each window.

That score can help prioritize human review if it is accurate and timely enough. It does not make a maintenance decision by itself. In this project, the test results show that this model is not accurate enough to drive work orders.

## 6. Which models are compared, and what do they do?

All three models receive the same 21 features and output a score for the positive class (failure reported within the next 24 hours).

| Model | Plain-language role |
|---|---|
| Logistic regression | A relatively simple weighted combination of sensor features; useful as an interpretable baseline. |
| Random forest | Many decision trees vote together; can capture nonlinear sensor interactions. |
| Histogram gradient boosting | Builds trees in sequence, with later trees correcting earlier errors. |

The training setup assigns greater weight to the rare failure class. This helps the models pay attention to failure examples, but can also increase false alarms.

The random forest had the highest mean PR-AUC over the two validation incidents, so it was selected. That means it ranked first among these three models on this small validation exercise; it does not mean it is a good or production-ready model.

## 7. How was the model evaluated?

### Chronological design

Randomly mixing adjacent time windows would let the model train on later data while being tested on earlier, nearly identical conditions. The project instead orders data by time:

1. Events 2 and 3 are used for forward event validation. For each event, the model trains only on data ending at least 48 hours before the event, then predicts the preceding seven days.
2. The model with the highest average precision across those two validation event windows is selected.
3. The F2-oriented alert threshold is selected from the pooled validation predictions. F2 gives recall more weight than precision.
4. The selected model is fit on development data ending 2020-07-08.
5. The later July period, including documented event 4 on 2020-07-15, is held out for the final reported test.

The validation set contains only two incidents and the final test effectively contains one reported incident. This is a very small evidence base. Treat its metrics as a case study result, not as a stable estimate of future performance.

### Metrics and what each means

| Metric | Result | Interpretation |
|---|---:|---|
| PR-AUC | 0.0125 | Overall ranking quality for rare positive windows. Low; approximately the positive-window prevalence in the holdout (103 / 6,689 ≈ 0.015), so ranking is weak. |
| ROC-AUC | 0.439 | Probability that a randomly chosen positive window receives a higher score than a negative. 0.5 is chance ranking; this result is below that reference. |
| Recall | 81.6% | 84 of 103 labeled pre-failure windows were flagged; 19 were missed. |
| Precision | 1.67% | 84 of 5,027 flagged windows were positive; 4,943 were outside the labeled pre-failure window. About 60 non-positive flagged windows per positive flagged window. |
| F1 | 0.0327 | Combined precision/recall score that weights them equally. Low because precision is very low. |
| F2 | 0.0772 | F-score that weights recall more heavily. Still low on the held-out period. |
| Balanced accuracy | 0.533 | Average of sensitivity and specificity; only modestly above 0.5. |
| Brier score | 0.0212 | Mean squared probability error; lower is better. Read alongside the low event prevalence and other metrics. |

The selected probability cutoff was 0.02. The threshold was chosen to favor recall on validation incidents. At this cutoff the holdout confusion counts are:

| | Predicted alert | No alert |
|---|---:|---:|
| Labeled pre-failure window | 84 | 19 |
| Other window | 4,943 | 1,643 |

These are ten-minute windows, not unique failures, trains, or work orders. Consecutive alarms may belong to one continuous episode. Precision’s “false positive” label means outside the documented 24-hour window; it does not prove the equipment needed no maintenance.

The first above-threshold score in the held-out period was on 2020-07-08 at 15:40, around 167 hours before the documented failure. Since this is far earlier than the 24-hour target, it may be persistent background alerting rather than a useful actionable warning. It must not be presented as a seven-day successful warning without event-level alarm analysis.

## 8. What is the business decision?

**Current decision: do not automate maintenance actions from this model.** The current alert cutoff produces a large number of windows requiring review and has weak ranking performance. One held-out incident cannot demonstrate avoided downtime, cost savings, or reliable future detection.

The analysis can still guide a low-risk learning process:

1. **Group alerts into episodes.** For example, merge adjacent flagged 10-minute windows and require a quiet period before opening a new episode. Choose the gap with maintenance engineers.
2. **Review alerts in shadow mode.** Show alert episodes to engineers without changing planned maintenance. Record whether they see actionable symptoms.
3. **Capture outcomes.** Record inspections, confirmed defects, repair work, time-to-failure, and alerts judged non-actionable.
4. **Compare with current practice.** Measure whether the approach identifies issues earlier or reduces disruption compared with existing scheduled or condition-based checks.
5. **Set an operating threshold using costs.** Estimate the cost of an inspection, missed failure, false alarm, downtime hour, and repair. Choose a threshold that reflects that cost trade-off rather than maximizing a generic score.
6. **Validate on more incidents and assets.** A prospective test across more compressors and failure events is needed before a controlled pilot.

## 9. How could a company estimate financial value?

For each alert episode, define whether it would trigger inspection. Then estimate:

**Net value = avoided failure/downtime cost − inspection cost − intervention cost − costs of unnecessary alerts.**

This project dataset does not contain those cost values, maintenance actions, or counterfactual outcomes. Therefore, it cannot currently estimate savings. A company would need approximate costs and prospective records to build a credible business case.

## 10. Reproducing the analysis

From the repository root in PowerShell:

```powershell
.venv\Scripts\python.exe train.py
.venv\Scripts\python.exe app.py
```

Open `http://127.0.0.1:8501` for the interactive report. Move the alert-cutoff slider to see its impact on the held-out window counts, precision, and recall. This slider is for understanding the trade-off; the final holdout must not be used to tune the deployed threshold.

The pipeline code is in `src/data.py`, `src/features.py`, and `src/evaluate.py`. The report dashboard is in `app.py`. Generated metrics are in `artifacts/metrics.json` and `artifacts/cv_results.csv`.

## References

- UCI Machine Learning Repository, [MetroPT-3 Dataset](https://archive.ics.uci.edu/dataset/791/metropt%2B3%2B).
- Davari, N., Veloso, B., Ribeiro, R., Pereira, P. M., & Gama, J. (2021). “Predictive maintenance based on anomaly detection using deep learning for air production unit in the railway industry.” DSAA.
- Veloso, B., Ribeiro, R. P., Gama, J., & Pereira, P. M. (2022). “The MetroPT dataset for predictive maintenance.” *Scientific Data*, 9, 764.
