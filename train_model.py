"""Train a weekly SARIMA baseline and a leakage-free residual correction model."""
import json

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

from data_preprocessing import ROOT, ENV_COLUMNS, FEATURE_COLUMNS, load_data, clean_data, future_features


def fit_sarima(train):
    candidates = [((1, 0, 1), (1, 0, 0, 7)), ((1, 1, 1), (1, 0, 0, 7))]
    fits, rows = [], []
    for order, seasonal_order in candidates:
        result = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                         trend='c').fit(disp=False, maxiter=250)
        converged = bool(result.mle_retvals['converged'])
        rows.append({'order': str(order), 'seasonal_order': str(seasonal_order),
                     'aic': result.aic, 'converged': converged})
        if converged and np.isfinite(result.aic):
            fits.append(result)
    if not fits:
        raise RuntimeError('Neither SARIMA candidate converged. Inspect data and fit warnings.')
    return min(fits, key=lambda model: model.aic), pd.DataFrame(rows)


def add_alerts(forecast, threshold):
    forecast = forecast.copy()
    forecast['surge_status'] = np.where(forecast['hybrid'] > threshold, 'High Risk', 'Normal')
    return forecast


def forecast_next_7_days(history, sarima, residual_model=None, threshold=80, days=7):
    features = future_features(history, days)
    baseline = np.maximum(0, np.asarray(sarima.forecast(days)))
    correction = np.zeros(days) if residual_model is None else residual_model.predict(features)
    forecast = pd.DataFrame({'date': features.index, 'sarima': baseline,
                             'residual_correction': correction,
                             'hybrid': np.maximum(0, baseline + correction)})
    return add_alerts(forecast, threshold)


def rolling_backtest(df, start, end, sarima, residual_model=None):
    """Issue each whole horizon before revealing its outcomes or predictors."""
    forecasts, feature_blocks = [], []
    state = sarima
    for origin in range(start, end, 7):
        days = min(7, end - origin)
        history = df.iloc[:origin]
        features = future_features(history, days)
        forecast = forecast_next_7_days(history, state, residual_model, days=days)
        forecast['origin'] = history.index[-1]
        forecast['horizon'] = np.arange(1, days + 1)
        forecast['naive'] = history['admissions'].ffill().fillna(0).iloc[-7:].to_numpy()[:days]
        # Outcomes enter only after every prediction in this block has been issued.
        actual = df['admissions'].iloc[origin:origin + days]
        forecast['actual'] = actual.to_numpy()
        forecast['observed'] = actual.notna().to_numpy()
        forecasts.append(forecast)
        feature_blocks.append(features)
        state = state.append(actual, refit=False)
    return pd.concat(forecasts, ignore_index=True), pd.concat(feature_blocks), state


def evaluate(predictions):
    observed = predictions.dropna(subset=['actual'])
    if observed.empty:
        raise ValueError('No observed admissions available for evaluation.')
    rows = []
    for name in ['naive', 'sarima', 'hybrid']:
        rmse = float(np.sqrt(mean_squared_error(observed['actual'], observed[name])))
        mean = float(observed['actual'].mean())
        rows.append({'model': name, 'MAE': mean_absolute_error(observed['actual'], observed[name]),
                     'RMSE': rmse, 'NRMSE': rmse / mean if mean > 0 else np.nan,
                     'n_scored': len(observed)})
    return pd.DataFrame(rows).set_index('model')


def surge_precision(predictions, threshold):
    observed = predictions.dropna(subset=['actual'])
    alerts = observed['hybrid'] > threshold
    count = int(alerts.sum())
    true_positives = int((alerts & (observed['actual'] > threshold)).sum())
    return {'threshold': threshold, 'predicted_alerts': count, 'true_positives': true_positives,
            'actual_surges': int((observed['actual'] > threshold).sum()),
            'precision': true_positives / count if count else None}


def make_plots(df, predictions, folder):
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    df['admissions'].plot(ax=axes[0, 0], linewidth=0.7, title='Daily simulated admissions')
    axes[0, 0].set_ylabel('Admissions / day')
    df[['aqi', 'pm25']].plot(ax=axes[0, 1], linewidth=0.7, title='Environmental trends')
    axes[0, 1].set_ylabel('AQI sub-index / PM2.5 (ug/m3)')
    axes[0, 1].legend(['Derived PM2.5 AQI', 'PM2.5 (ug/m3)'])
    axes[1, 0].scatter(df['pm25'], df['admissions'], alpha=0.3, s=10)
    axes[1, 0].set(xlabel='Same-day PM2.5 (ug/m3)', ylabel='Simulated admissions',
                    title='Association, not causation')
    df['admissions'].resample('MS').mean().plot(ax=axes[1, 1], title='Monthly mean admissions')
    axes[1, 1].set_ylabel('Admissions / day')
    df.groupby(df.index.dayofweek)['admissions'].mean().plot.bar(ax=axes[2, 0], title='Mean by weekday')
    axes[2, 0].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], rotation=0)
    axes[2, 0].set(xlabel='Day of week', ylabel='Admissions / day')
    columns = ['admissions'] + ENV_COLUMNS
    correlation = df[columns].corr()
    chart = axes[2, 1].imshow(correlation, vmin=-1, vmax=1, cmap='coolwarm')
    axes[2, 1].set_xticks(range(len(columns)), columns, rotation=35, ha='right')
    axes[2, 1].set_yticks(range(len(columns)), columns)
    axes[2, 1].set_title('Pearson correlation (simulated outcomes)')
    fig.colorbar(chart, ax=axes[2, 1], shrink=0.75)
    fig.suptitle('Demonstration data: all hospital admission counts are simulated')
    fig.tight_layout()
    fig.savefig(folder / 'eda.png', dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 4))
    predictions.set_index('date')[['actual', 'sarima', 'hybrid']].plot(ax=ax, linewidth=1)
    ax.set(title='Held-out seven-day backtest: simulated admissions', ylabel='Admissions / day', xlabel='Date')
    fig.tight_layout()
    fig.savefig(folder / 'test_predictions.png', dpi=140)
    plt.close(fig)


def train_project():
    if not (ROOT / 'data/hospital_air_quality.csv').exists():
        from generate_demo_data import generate_demo
        generate_demo()
    df, audit = clean_data(load_data())
    if len(df) < 365 or df['admissions'].notna().sum() < 300:
        raise ValueError('Use at least one year of data and 300 observed outcomes.')
    model_folder, results_folder = ROOT / 'models', ROOT / 'results'
    model_folder.mkdir(exist_ok=True)
    results_folder.mkdir(exist_ok=True)
    train_end, validation_end = int(0.70 * len(df)), int(0.85 * len(df))
    print('Fitting two SARIMA candidates on the training period...', flush=True)
    sarima, candidates = fit_sarima(df['admissions'].iloc[:train_end])
    candidates.to_csv(results_folder / 'sarima_candidates.csv', index=False)
    print('Generating out-of-sample validation residuals in seven-day blocks...', flush=True)
    validation, features, validation_state = rolling_backtest(df, train_end, validation_end, sarima)
    residuals = validation['actual'] - validation['sarima']
    valid = residuals.notna().to_numpy()
    residual_model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05,
                                  objective='reg:squarederror', random_state=42, n_jobs=1)
    residual_model.fit(features.iloc[valid], residuals[valid])
    print('Evaluating on the untouched test segment...', flush=True)
    test, _, final_state = rolling_backtest(df, validation_end, len(df), validation_state, residual_model)
    metrics = evaluate(test)
    rmse_baseline = float(metrics.loc['sarima', 'RMSE'])
    improvement = 100 * (rmse_baseline - float(metrics.loc['hybrid', 'RMSE'])) / rmse_baseline if rmse_baseline > 0 else None
    provenance = json.loads((ROOT / 'data/provenance.json').read_text(encoding='utf-8'))
    summary = {'data_status': provenance, 'evaluation': 'Simulated outcomes; rolling seven-day blocks',
               'split_rows': {'train': train_end, 'validation': validation_end-train_end, 'test': len(df)-validation_end},
               'train_last_date': str(df.index[train_end-1].date()),
               'validation_last_date': str(df.index[validation_end-1].date()),
               'test_first_date': str(df.index[validation_end].date()),
               'test_last_date': str(df.index[-1].date()),
               'test_origins': int(test['origin'].nunique()),
               'full_seven_day_test_origins': int((test.groupby('origin').size() == 7).sum()),
               'sarima_order': list(sarima.model.order), 'seasonal_order': list(sarima.model.seasonal_order),
               'sarima_to_hybrid_improvement_percent': improvement,
               'surge_at_80': surge_precision(test, 80), 'features': FEATURE_COLUMNS}
    test.to_csv(results_folder / 'test_predictions.csv', index=False)
    validation.to_csv(results_folder / 'validation_predictions.csv', index=False)
    features.to_csv(results_folder / 'validation_features.csv', index_label='date')
    metrics.to_csv(results_folder / 'metrics.csv')
    horizon_metrics = []
    for horizon, rows in test.groupby('horizon'):
        table = evaluate(rows).reset_index()
        table.insert(0, 'horizon', horizon)
        horizon_metrics.append(table)
    pd.concat(horizon_metrics).to_csv(results_folder / 'metrics_by_horizon.csv', index=False)
    (results_folder / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (results_folder / 'cleaning_report.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    joblib.dump({'sarima': final_state, 'residual_model': residual_model,
                 'history': df, 'summary': summary}, model_folder / 'forecast_models.joblib', compress=3)
    future = forecast_next_7_days(df, final_state, residual_model)
    future.to_csv(results_folder / 'next_7_days.csv', index=False)
    make_plots(df, test, results_folder)
    print(metrics.round(4).to_string())
    print('SARIMA -> hybrid RMSE improvement (%):', improvement)
    print('Surge precision at 80:', summary['surge_at_80'])
    print('Saved models, plots and results. All admission outcomes are SIMULATED.')


if __name__ == '__main__':
    train_project()
