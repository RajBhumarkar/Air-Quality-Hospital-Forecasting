"""Create explicitly simulated admissions; use bundled UCI data when available."""
import argparse
import io
import json
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from data_preprocessing import ROOT, clean_data

UCI_URL = 'https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip'


def pm25_aqi(pm25):
    """US PM2.5-only sub-index, not official Beijing or multi-pollutant AQI."""
    concentration = np.floor(np.asarray(pm25, dtype=float) * 10) / 10
    result = np.full(concentration.shape, np.nan)
    bands = [(0, 9, 0, 50), (9.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
             (55.5, 125.4, 151, 200), (125.5, 225.4, 201, 300),
             (225.5, 325.4, 301, 500)]
    for low, high, index_low, index_high in bands:
        mask = (concentration >= low) & (concentration <= high)
        if index_high == 500:
            mask = concentration >= low
        result[mask] = index_low + (concentration[mask] - low) * (index_high-index_low)/(high-low)
    return np.floor(result + 0.5)


def download_environment(archive_path=None):
    if archive_path:
        archive = zipfile.ZipFile(archive_path)
    else:
        with urllib.request.urlopen(UCI_URL, timeout=60) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
    names = archive.namelist()
    if not any('Aotizhongxin' in name for name in names):
        nested = next(name for name in names if name.endswith('.zip'))
        archive = zipfile.ZipFile(io.BytesIO(archive.read(nested)))
    name = next(name for name in archive.namelist() if 'Aotizhongxin' in name and name.endswith('.csv'))
    hourly = pd.read_csv(archive.open(name))
    hourly['date'] = pd.to_datetime(hourly[['year', 'month', 'day']])
    hourly['humidity'] = (100 * np.exp(17.625 * hourly['DEWP'] / (243.04 + hourly['DEWP'])
                            - 17.625 * hourly['TEMP'] / (243.04 + hourly['TEMP']))).clip(0, 100)
    hourly = hourly.rename(columns={'PM2.5': 'pm25', 'TEMP': 'temperature'})
    columns = ['pm25', 'temperature', 'humidity']
    grouped = hourly.groupby('date')[columns]
    daily = grouped.mean().where(grouped.count() >= 18)
    daily = daily.loc['2014-01-01':'2016-12-31'].copy()
    daily['aqi'] = pm25_aqi(daily['pm25'])
    daily['city'] = 'Beijing-Aotizhongxin-demo'
    daily['environment_source'] = 'UCI observed; humidity and PM2.5 AQI derived'
    daily.to_csv(ROOT / 'data/environment_observed.csv', index_label='date')
    return daily.reset_index()


def synthetic_environment():
    rng = np.random.default_rng(42)
    dates = pd.date_range('2014-01-01', '2016-12-31', freq='D')
    winter = np.cos(2 * np.pi * (dates.dayofyear.to_numpy() - 15) / 365.25)
    pollution = np.zeros(len(dates))
    for i in range(1, len(dates)):
        pollution[i] = 0.75 * pollution[i-1] + rng.normal(0, 15)
    df = pd.DataFrame({'date': dates, 'city': 'Synthetic-city-demo'})
    df['pm25'] = np.clip(65 + 25 * winter + pollution, 3, 400)
    df['temperature'] = 14 - 14 * winter + rng.normal(0, 3, len(df))
    df['humidity'] = np.clip(55 - 12 * winter + rng.normal(0, 8, len(df)), 5, 100)
    df['aqi'] = pm25_aqi(df['pm25'])
    df['environment_source'] = 'SYNTHETIC environment'
    return df


def generate_demo(download=False, archive_path=None, synthetic=False):
    (ROOT / 'data').mkdir(exist_ok=True)
    observed_path = ROOT / 'data/environment_observed.csv'
    if synthetic:
        environment = synthetic_environment()
    elif download or archive_path:
        environment = download_environment(archive_path)
    elif observed_path.exists():
        environment = pd.read_csv(observed_path, parse_dates=['date'])
    else:
        environment = synthetic_environment()
    environment['date'] = pd.to_datetime(environment['date'])
    environment['admissions'] = np.nan
    cleaned, _ = clean_data(environment)
    rng = np.random.default_rng(2026)
    winter = np.cos(2 * np.pi * (cleaned.index.dayofyear.to_numpy() - 15) / 365.25)
    exposure = cleaned['pm25'].shift(1).rolling(3, min_periods=1).mean().fillna(35)
    cold = (15 - cleaned['temperature']).clip(lower=0)
    weekly = np.array([6, 3, 1, 0, 1, -4, -5])[cleaned.index.dayofweek]
    # Illustrative coefficients, not estimates of medical effects.
    expected = 49 + 8 * winter + weekly + 0.09 * exposure + 0.35 * cold
    labels = pd.DataFrame({'date': cleaned.index, 'admissions': rng.poisson(expected.clip(lower=5)),
                           'admissions_source': 'SIMULATED - not hospital observations'})
    labels.to_csv(ROOT / 'data/admissions_simulated.csv', index=False)
    merged = environment.drop(columns='admissions').merge(labels, on='date', validate='one_to_one')
    merged.to_csv(ROOT / 'data/hospital_air_quality.csv', index=False)
    metadata = {'admissions': 'SIMULATED; no real hospital outcomes',
                'environment': environment['environment_source'].iloc[0],
                'rows': len(merged), 'start': str(cleaned.index[0].date()),
                'end': str(cleaned.index[-1].date()), 'admission_seed': 2026,
                'source_url': UCI_URL if not synthetic and observed_path.exists() else None}
    (ROOT / 'data/provenance.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', help='Refresh UCI environmental data (internet required).')
    parser.add_argument('--archive', help='Use a local UCI ZIP instead of downloading.')
    parser.add_argument('--synthetic', action='store_true', help='Use fully synthetic environmental data as well.')
    args = parser.parse_args()
    generate_demo(args.download, args.archive, args.synthetic)
