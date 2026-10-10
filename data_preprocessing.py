"""Simple daily cleaning and forecast-time features."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ENV_COLUMNS = ['pm25', 'aqi', 'temperature', 'humidity']
FEATURE_COLUMNS = [
    'day_of_week', 'month', 'weekend', 'pm25_lag1', 'pm25_lag3',
    'pm25_previous7', 'aqi_lag1', 'temperature', 'humidity',
    'admissions_lag7', 'horizon',
]


def load_data(file_path=ROOT / 'data/hospital_air_quality.csv'):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'], errors='raise')
    return df


def clean_data(df):
    """Return one daily series and an audit; never invent admission targets."""
    df = df.copy()
    required = ['date', 'city', 'admissions'] + ENV_COLUMNS
    if not set(required).issubset(df.columns):
        raise ValueError('Required columns: ' + ', '.join(required))
    df['date'] = pd.to_datetime(df['date'], errors='raise').dt.normalize()
    if df['date'].isna().any() or df['city'].isna().any():
        raise ValueError('Date and city must be present.')
    if df['city'].nunique() != 1:
        raise ValueError('Select exactly one city before cleaning; do not mix series.')
    report = {'input_rows': len(df), 'missing_before': df[required].isna().sum().to_dict()}
    report['duplicates_removed'] = int(df.duplicated(['date', 'city']).sum())
    df = df.drop_duplicates(['date', 'city'], keep='first').sort_values('date')
    city = df['city'].iloc[0]
    df = df.set_index('date')
    dates = pd.date_range(df.index.min(), df.index.max(), freq='D', name='date')
    report['missing_dates_added'] = len(dates) - len(df)
    df = df.reindex(dates)
    df['city'] = city
    ranges = {'pm25': (0, 1500), 'aqi': (0, 3000),
              'temperature': (-60, 60), 'humidity': (0, 100), 'admissions': (0, np.inf)}
    report['invalid_values'] = {}
    for column, (low, high) in ranges.items():
        df[column] = pd.to_numeric(df[column], errors='coerce')
        valid = df[column].between(low, high) & np.isfinite(df[column])
        if column == 'admissions':
            valid = valid & (df[column] % 1 == 0)
        report['invalid_values'][column] = int((df[column].notna() & ~valid).sum())
        df.loc[~valid, column] = np.nan
    df['admissions_observed'] = df['admissions'].notna()
    # Forward-fill predictors only. Fixed defaults handle leading gaps without future data.
    report['environment_values_filled'] = df[ENV_COLUMNS].isna().sum().to_dict()
    defaults = {'pm25': 35, 'aqi': 100, 'temperature': 15, 'humidity': 50}
    df[ENV_COLUMNS] = df[ENV_COLUMNS].ffill().fillna(defaults)
    report['admissions_missing_retained'] = int(df['admissions'].isna().sum())
    report['daily_rows'] = len(df)
    return df, report


def create_features(df):
    """Shift before rolling. For forecasting, df must contain assumed future predictors."""
    features = pd.DataFrame(index=df.index)
    features['day_of_week'] = df.index.dayofweek
    features['month'] = df.index.month
    features['weekend'] = (df.index.dayofweek >= 5).astype(int)
    features['pm25_lag1'] = df['pm25'].shift(1)
    features['pm25_lag3'] = df['pm25'].shift(3)
    features['pm25_previous7'] = df['pm25'].shift(1).rolling(7).mean()
    features['aqi_lag1'] = df['aqi'].shift(1)
    features['temperature'] = df['temperature']
    features['humidity'] = df['humidity']
    # This feature stays historical for every day of a seven-day horizon.
    features['admissions_lag7'] = df['admissions'].ffill().fillna(0).shift(7)
    return features


def future_features(history, days=7):
    """Only accepts history. Unknown pollution/weather use the last seven-day mean."""
    if not 1 <= days <= 7 or len(history) < 7:
        raise ValueError('Need at least seven historical days and a horizon from 1 to 7.')
    dates = pd.date_range(history.index[-1] + pd.Timedelta(days=1), periods=days, freq='D')
    future = pd.DataFrame(index=dates, columns=ENV_COLUMNS + ['admissions'], dtype=float)
    future[ENV_COLUMNS] = history[ENV_COLUMNS].tail(7).mean().to_numpy()
    combined = pd.concat([history[ENV_COLUMNS + ['admissions']], future])
    features = create_features(combined).tail(days).copy()
    features['horizon'] = np.arange(1, days + 1)
    return features[FEATURE_COLUMNS]
