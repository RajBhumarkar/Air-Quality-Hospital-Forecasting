"""Small executable checks for the forecasting and dashboard requirements."""
import json

import joblib
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from data_preprocessing import ROOT, ENV_COLUMNS, load_data, clean_data, create_features, future_features
from generate_demo_data import synthetic_environment, pm25_aqi
from train_model import forecast_next_7_days, rolling_backtest, evaluate, surge_precision, add_alerts


def run_checks():
    checks = []
    raw = load_data()
    df, audit = clean_data(raw)
    assert len(df) == 1096 and df.index.is_monotonic_increasing and df.index.freqstr == 'D'
    assert not df[ENV_COLUMNS].isna().any().any()
    checks.append('CSV loads as 1,096 ordered daily observations; environmental gaps filled.')

    dirty = raw.head(20).copy()
    dirty.loc[0, 'pm25'] = np.nan
    dirty.loc[4, 'humidity'] = 150
    dirty.loc[5, 'admissions'] = -2
    dirty = pd.concat([dirty.drop(index=8), dirty.iloc[[2]]], ignore_index=True)
    cleaned, report = clean_data(dirty.sample(frac=1, random_state=1))
    assert report['duplicates_removed'] == 1 and report['missing_dates_added'] == 1
    assert cleaned.iloc[0]['pm25'] == 35
    assert cleaned.iloc[4]['humidity'] == cleaned.iloc[3]['humidity']
    assert pd.isna(cleaned.iloc[5]['admissions']) and pd.isna(cleaned.iloc[8]['admissions'])
    assert not cleaned.iloc[5]['admissions_observed']
    checks.append('Missing/invalid predictors, leading gaps, duplicates and missing dates handled; missing targets preserved.')

    altered = raw.copy()
    altered.loc[100:, ENV_COLUMNS + ['admissions']] = 999
    changed, _ = clean_data(altered)
    pd.testing.assert_frame_equal(df[ENV_COLUMNS + ['admissions']].iloc[:100],
                                  changed[ENV_COLUMNS + ['admissions']].iloc[:100])
    features = create_features(df)
    assert features.iloc[20]['pm25_previous7'] == df['pm25'].iloc[13:20].mean()
    assert features.iloc[20]['pm25_lag1'] == df['pm25'].iloc[19]
    assert features.iloc[20]['pm25_lag3'] == df['pm25'].iloc[17]
    assert features.iloc[20]['aqi_lag1'] == df['aqi'].iloc[19]
    future = future_features(df.iloc[:100])
    np.testing.assert_allclose(future['admissions_lag7'], df['admissions'].iloc[93:100])
    np.testing.assert_allclose(future['temperature'], df['temperature'].iloc[93:100].mean())
    checks.append('Preprocessing prefix invariance, shifted lags, previous-seven-day rolling mean and forecast-time assumptions verified.')

    fallback = synthetic_environment()
    assert len(fallback) == 1096 and fallback['environment_source'].str.contains('SYNTHETIC').all()
    assert not fallback[ENV_COLUMNS].isna().any().any()
    np.testing.assert_equal(pm25_aqi([0, 9, 9.1, 35.4, 225.5, 325.4]), [0, 50, 51, 100, 301, 500])
    checks.append('Offline synthetic environmental fallback and AQI breakpoint calculations verified.')

    bundle = joblib.load(ROOT / 'models/forecast_models.joblib')
    sarima, model = bundle['sarima'], bundle['residual_model']
    candidates = pd.read_csv(ROOT / 'results/sarima_candidates.csv')
    assert candidates['converged'].all() and model.n_features_in_ == 11
    assert np.isfinite(sarima.params).all()
    checks.append('Trained SARIMA converged; fitted XGBoost accepts 11 features; saved models reload.')
    forecast = forecast_next_7_days(df, sarima, model)
    assert len(forecast) == 7 and forecast['date'].min() == df.index[-1] + pd.Timedelta(days=1)
    np.testing.assert_allclose(forecast['hybrid'], np.maximum(0, forecast['sarima'] + model.predict(future_features(df))))
    assert (forecast[['sarima', 'hybrid']] >= 0).all().all()
    assert add_alerts(forecast, -1)['surge_status'].eq('High Risk').all()
    assert add_alerts(forecast, 1e6)['surge_status'].eq('Normal').all()
    exact = pd.DataFrame({'hybrid': [80., 80.01]})
    assert add_alerts(exact, 80)['surge_status'].tolist() == ['Normal', 'High Risk']
    checks.append('Seven-day dates, residual addition, nonnegative forecasts and strict threshold behavior verified.')

    test = pd.read_csv(ROOT / 'results/test_predictions.csv', parse_dates=['date', 'origin'])
    validation = pd.read_csv(ROOT / 'results/validation_predictions.csv', parse_dates=['date', 'origin'])
    assert validation['date'].max() < test['date'].min()
    assert (test['origin'] < test['date']).all()
    assert test['date'].is_unique and test.groupby('origin').size().max() == 7
    saved_metrics = pd.read_csv(ROOT / 'results/metrics.csv', index_col='model')
    np.testing.assert_allclose(evaluate(test)[['MAE', 'RMSE', 'NRMSE']], saved_metrics[['MAE', 'RMSE', 'NRMSE']])
    assert surge_precision(test, 1e6)['precision'] is None
    example = pd.DataFrame({'actual': [100, 70, np.nan], 'hybrid': [100, 100, 100]})
    assert surge_precision(example, 80)['precision'] == 0.5
    checks.append('Chronological disjoint validation/test, metric recomputation and precision missing-label/no-alert cases verified.')

    # An adversarial test changes ALL within-horizon measurements and targets.
    # Predictions at the first test origin must remain identical.
    start = bundle['summary']['split_rows']['train'] + bundle['summary']['split_rows']['validation']
    origin_state = sarima.apply(df['admissions'].iloc[:start], refit=False)
    original, original_features, _ = rolling_backtest(df, start, start + 7, origin_state, model)
    corrupted = df.copy()
    corrupted.iloc[start:start+7, corrupted.columns.get_indexer(ENV_COLUMNS + ['admissions'])] = 999
    other, other_features, _ = rolling_backtest(corrupted, start, start + 7, origin_state, model)
    pd.testing.assert_frame_equal(original_features, other_features)
    np.testing.assert_allclose(original[['naive', 'sarima', 'hybrid']], other[['naive', 'sarima', 'hybrid']])
    checks.append('Adversarial seven-day leakage check: changing every future predictor/outcome leaves all horizon forecasts unchanged.')

    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=180).run()
    assert not app.exception, str(app.exception)
    assert len(app.slider) == 1 and len(app.metric) >= 7
    app.slider[0].set_value(0).run()
    assert not app.exception
    count_low = next(item.value for item in app.metric if item.label == 'Predicted high-risk days')
    app.slider[0].set_value(200).run()
    count_high = next(item.value for item in app.metric if item.label == 'Predicted high-risk days')
    assert not app.exception and int(count_low) >= int(count_high)
    assert int(count_low) == 7 and int(count_high) == 0
    checks.append('Streamlit AppTest renders dashboard and slider changes high-risk count from seven to zero without exceptions.')
    report = {'status': 'passed', 'checks_executed': checks}
    (ROOT / 'results/checks.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for item in checks:
        print('PASS:', item)


if __name__ == '__main__':
    run_checks()
