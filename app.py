"""Run with: python -m streamlit run app.py"""
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from data_preprocessing import ROOT, ENV_COLUMNS
from train_model import forecast_next_7_days, surge_precision

st.set_page_config(page_title='Air Quality & Admissions', layout='wide')
st.title('Air Quality-Driven Hospital Admission Forecasting')
st.write('A student project combining weekly SARIMA forecasts with XGBoost residual correction.')
st.warning('All hospital admission counts are SIMULATED. Academic demonstration only; not a medical decision system.')
model_path = ROOT / 'models/forecast_models.joblib'
if not model_path.exists():
    st.info('Run `python train_model.py` in the project folder, then refresh this page.')
    st.stop()


@st.cache_resource
def load_models(modified_time):
    # Load only the local artifact created by train_model.py; joblib is not safe for untrusted files.
    return joblib.load(model_path)


bundle = load_models(model_path.stat().st_mtime)
history, summary = bundle['history'], bundle['summary']
st.caption('Environment: ' + summary['data_status']['environment'])
st.caption(f"Historical demonstration: {history.index[0].date()} to {history.index[-1].date()}. "
           'One monitoring station paired with a hypothetical admission series; no real hospital is represented.')
with st.expander('Data source and assumptions'):
    st.write('Environmental source: [UCI Beijing Multi-Site Air Quality](https://doi.org/10.24432/C5RK5G), CC BY 4.0. '
             'Humidity is derived from temperature/dew point. AQI is a derived US PM2.5-only sub-index, not official Beijing AQI.')
    st.write('Missing environmental readings are forward-filled; leading gaps use fixed defaults. '
             'Missing admission outcomes stay missing and are excluded from accuracy and precision scoring.')

st.header('Historical analysis')
st.line_chart(history[['admissions']].rename(columns={'admissions': 'Simulated admissions / day'}))
left, right = st.columns(2)
with left:
    st.write('PM2.5 concentration (ug/m3)')
    st.line_chart(history[['pm25']])
with right:
    st.write('Derived PM2.5 AQI sub-index')
    st.line_chart(history[['aqi']])
with st.expander('Correlation and exploratory plots'):
    correlation = history[['admissions'] + ENV_COLUMNS].corr()
    fig, ax = plt.subplots(figsize=(6, 4))
    chart = ax.imshow(correlation, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(len(correlation)), correlation.columns, rotation=35, ha='right')
    ax.set_yticks(range(len(correlation)), correlation.columns)
    fig.colorbar(chart, ax=ax)
    st.pyplot(fig)
    plt.close(fig)
    st.caption('Correlation does not establish causation. The admissions relationship was specified in the simulation.')
    st.image(str(ROOT / 'results/eda.png'))

st.header('Model performance - simulated-data results')
metrics = pd.read_csv(ROOT / 'results/metrics.csv', index_col='model')
test = pd.read_csv(ROOT / 'results/test_predictions.csv', parse_dates=['date'])
columns = st.columns(3)
for column, model in zip(columns, ['naive', 'sarima', 'hybrid']):
    column.metric(model.title() + ' RMSE', f"{metrics.loc[model, 'RMSE']:.2f}")
left, right = st.columns(2)
left.metric('Hybrid NRMSE (RMSE / mean actual)', f"{metrics.loc['hybrid', 'NRMSE']:.3f}")
improvement = summary['sarima_to_hybrid_improvement_percent']
right.metric('RMSE improvement vs SARIMA', 'N/A' if improvement is None else f'{improvement:.2f}%')
st.caption(f"Chronological 70/15/15 split; {summary['test_origins']} test origins. "
           'Each block predicts up to seven days before observing any outcomes in that block. '
           'Negative improvement means the hybrid performed worse.')
st.dataframe(metrics.round(4))
st.line_chart(test.set_index('date')[['actual', 'sarima', 'hybrid']])
with st.expander('Accuracy by forecast horizon'):
    st.dataframe(pd.read_csv(ROOT / 'results/metrics_by_horizon.csv').round(4), hide_index=True)

st.header('Next seven days - demonstration forecast')
threshold = st.slider('Daily respiratory emergency-admission planning threshold', 0, 200, 80)
st.caption('A daily arrival planning threshold, not bed capacity. High Risk means the unrounded forecast is strictly greater than this threshold.')
forecast = forecast_next_7_days(history, bundle['sarima'], bundle['residual_model'], threshold)
st.write(f"Forecast dates follow the dataset's last date ({history.index[-1].date()}); these are not live forecasts for today.")
st.write('Unknown future PM2.5, AQI, temperature and humidity remain at their latest seven-day averages. '
         'No future measured values are used. Expected admissions are fractional counts.')
with st.expander('Environmental assumptions used for every forecast day'):
    st.dataframe(history[ENV_COLUMNS].tail(7).mean().rename('Assumed value').round(2))
st.metric('Predicted high-risk days', int((forecast['surge_status'] == 'High Risk').sum()))
st.dataframe(forecast.style.format({'sarima': '{:.2f}', 'hybrid': '{:.2f}', 'residual_correction': '{:.2f}'}), hide_index=True)
chart_data = forecast.set_index('date')[['sarima', 'hybrid']].copy()
chart_data['planning_threshold'] = threshold
st.line_chart(chart_data)
precision = surge_precision(test, threshold)
st.metric('Held-out surge-alert precision (simulated outcomes)',
          'N/A - no predicted alerts' if precision['precision'] is None else f"{precision['precision']:.1%}")
st.caption(f"Correct alerts: {precision['true_positives']} / {precision['predicted_alerts']}; "
           f"actual simulated surge days: {precision['actual_surges']}. Changing the threshold changes this exploratory metric; it is not test-set tuning.")
st.download_button('Download seven-day forecast CSV', forecast.to_csv(index=False),
                   'seven_day_simulated_admissions_forecast.csv', 'text/csv')
