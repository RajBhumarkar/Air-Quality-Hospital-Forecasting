# Verification record

Executed on 2026-10-09. All admission labels are simulated.

| Requested check | Executed evidence |
| --- | --- |
| Dataset loads | 1,096 ordered daily observations; 2014-01-01 through 2016-12-31 |
| Missing values | Bundled PM2.5/AQI gaps filled; injected leading gaps, invalid humidity, negative admissions, a duplicate and a missing date checked |
| Lag leakage | Lag-1/lag-3 values, shifted rolling mean, historical admission lag-7 and prefix invariance assertions passed |
| SARIMA training | Both candidates converged; training-only AIC selection recorded in `sarima_candidates.csv` |
| XGBoost training | 100-tree depth-3 residual regressor fitted to 164 validation residual targets; 11 features |
| Hybrid arithmetic | Compared saved model predictions against `max(0, SARIMA + residual prediction)` |
| Chronological evaluation | Disjoint validation/test dates, unique held-out dates and origin-before-target assertions; all metrics recomputed |
| Seven-day forecast | Seven consecutive dates after the history; all within-horizon observed environment/targets changed to 999 in an adversarial test without changing any forecast |
| Nonnegative predictions | SARIMA and hybrid output assertions passed |
| Surge behavior | Strict `>` boundary, threshold changes, missing actual labels and no-prediction precision cases checked |
| Streamlit application | AppTest rendered the complete script with no exceptions; slider 0 versus 200 changed the forecast high-risk count from 7 to 0 |
| Streamlit server | Started with the normal CLI; HTTP `/_stcore/health` returned status 200 and body `ok` from the same execution environment; process then stopped |
| README alignment | Script names, CLI arguments, data paths, saved artifact paths, split dates and reported metrics compared with implementation and generated results |

Additional executed checks: Python compilation of all five scripts; public UCI download and daily aggregation; end-to-end offline generator run in a separate scratch folder without any environmental source input. The latter produced 1,096 daily records, with synthetic environment and simulated admissions correctly labeled.

## Environment notes

Python 3.13, Pandas 2.2.3, NumPy 2.1.3, Matplotlib 3.9.2, statsmodels 0.14.6, XGBoost 3.4.1, scikit-learn 1.6.1, Streamlit 1.55.0, joblib 1.4.2.

An initial test assertion tried to read optimizer metadata from a state-updated SARIMA result. Those results do not retain `mle_retvals`; the assertion was corrected to check the saved original candidate convergence records and finite model parameters. Forecasting code was unchanged.

The preinstalled testing environment had an incomplete MarkupSafe package, a transitive dependency of Streamlit's chart rendering. It was repaired in a **workspace-only temporary dependency folder** with the official PyPI MarkupSafe 3.0.3 wheel, then the complete test script passed. No forecasting/chart/widget behavior was mocked or disabled. An initial 60-second AppTest startup timeout was increased to 180 seconds because stack inspection was slow in the sandbox. A clean end-to-end installation of all requirements in a new virtual environment was not executed successfully here: this sandbox's standard Python `venv` attempt failed during `ensurepip`. Recommended setup commands target a normal Python installation; the directly executed training and app commands used the existing packages plus the isolated dependency repair.

## Dashboard screenshots / browser limitation

The in-app browser timed out connecting to the sandboxed loopback server. A browser-level visual inspection and dashboard screenshots therefore **could not be completed**. This is separate from the successful same-environment HTTP health check and Streamlit AppTest rendering/slider tests. No dashboard screenshot is fabricated.

`eda.png` and `test_predictions.png` are generated Matplotlib plots; both were opened and visually inspected. They are not browser screenshots. To add real dashboard screenshots after starting the app locally, capture the model-performance and seven-day-forecast sections and place them here.

The exact passing assertions are listed in `checks.json`. Server launch evidence is in `server_check.json`. Tests do not establish clinical validity, robustness on real hospitals or complete cross-platform compatibility.
