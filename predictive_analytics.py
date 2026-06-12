"""
Predictive Analytics Using Historical Data
Thiranex - Skill Development & Future Tech
Project 3

Features:
- 10 years of synthetic monthly sales data
- Data cleaning & preprocessing (missing values, feature engineering)
- Linear Regression model
- Polynomial Regression for non-linear trend
- Future 12-month forecast
- Evaluation: MAE, RMSE, R²
- 4-panel visualization report
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# 1. GENERATE HISTORICAL DATASET
# ─────────────────────────────────────────
np.random.seed(42)
n = 120  # 10 years monthly

dates = pd.date_range(start='2015-01-01', periods=n, freq='MS')
trend      = np.linspace(50000, 150000, n)
seasonality = 15000 * np.sin(2 * np.pi * np.arange(n) / 12)
noise      = np.random.normal(0, 4000, n)
sales      = trend + seasonality + noise

df = pd.DataFrame({
    'date': dates,
    'sales': sales.round(2),
    'month': dates.month,
    'year': dates.year,
    'quarter': dates.quarter,
})

# Inject missing values
df.loc[[5, 23, 67], 'sales'] = np.nan
print(f"Missing values before cleaning: {df['sales'].isna().sum()}")

# ─────────────────────────────────────────
# 2. CLEANING & FEATURE ENGINEERING
# ─────────────────────────────────────────
df['sales'] = df['sales'].interpolate(method='linear')
print(f"Missing values after cleaning : {df['sales'].isna().sum()}")

df['time_index'] = np.arange(n)
df['month_sin']  = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos']  = np.cos(2 * np.pi * df['month'] / 12)

# ─────────────────────────────────────────
# 3. TRAIN / TEST SPLIT  (80 / 20)
# ─────────────────────────────────────────
split = int(n * 0.8)
train_df, test_df = df.iloc[:split], df.iloc[split:]

features = ['time_index', 'month_sin', 'month_cos', 'quarter']
X_train, y_train = train_df[features], train_df['sales']
X_test,  y_test  = test_df[features],  test_df['sales']

# ─────────────────────────────────────────
# 4. MODEL A — LINEAR REGRESSION
# ─────────────────────────────────────────
lr = make_pipeline(StandardScaler(), LinearRegression())
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

mae_lr  = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr   = r2_score(y_test, y_pred_lr)

print(f"\n── Linear Regression ───────────────────────")
print(f"MAE  : ₹{mae_lr:,.0f}   RMSE: ₹{rmse_lr:,.0f}   R²: {r2_lr:.4f}")

# ─────────────────────────────────────────
# 5. MODEL B — POLYNOMIAL REGRESSION (deg 3)
# ─────────────────────────────────────────
poly = make_pipeline(PolynomialFeatures(degree=3), StandardScaler(), LinearRegression())
poly.fit(X_train, y_train)
y_pred_poly = poly.predict(X_test)

mae_poly  = mean_absolute_error(y_test, y_pred_poly)
rmse_poly = np.sqrt(mean_squared_error(y_test, y_pred_poly))
r2_poly   = r2_score(y_test, y_pred_poly)

print(f"\n── Polynomial Regression (deg=3) ───────────")
print(f"MAE  : ₹{mae_poly:,.0f}   RMSE: ₹{rmse_poly:,.0f}   R²: {r2_poly:.4f}")

# ─────────────────────────────────────────
# 6. FUTURE 12-MONTH FORECAST
# ─────────────────────────────────────────
future_dates = pd.date_range(start=df['date'].max() + pd.offsets.MonthBegin(),
                              periods=12, freq='MS')
future_df = pd.DataFrame({
    'time_index': np.arange(n, n + 12),
    'month':      future_dates.month,
    'quarter':    future_dates.quarter,
})
future_df['month_sin'] = np.sin(2 * np.pi * future_df['month'] / 12)
future_df['month_cos'] = np.cos(2 * np.pi * future_df['month'] / 12)

# Use the better model (poly) for forecast
forecast = poly.predict(future_df[features])

# ─────────────────────────────────────────
# 7. VISUALIZATIONS (4-panel)
# ─────────────────────────────────────────
BLUE   = '#4361EE'
PINK   = '#F72585'
GREEN  = '#06D6A0'
PURPLE = '#7209B7'
ORANGE = '#FB8500'

fig = plt.figure(figsize=(15, 11))
fig.patch.set_facecolor('#F8F9FA')
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# A – Full historical trend
ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor('#FFFFFF')
ax1.plot(df['date'], df['sales'], color=BLUE, lw=1.8, label='Historical Sales')
ax1.fill_between(df['date'], df['sales'], alpha=0.1, color=BLUE)
ax1.axvline(df['date'].iloc[split], color=ORANGE, lw=1.5, linestyle='--', label='Train/Test split')
ax1.set_title('A  Historical Sales Trend — 10 Years Monthly Data', fontweight='bold', fontsize=12)
ax1.set_ylabel('Sales (₹)')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'₹{x/1000:.0f}K'))
ax1.legend(); ax1.grid(axis='y', alpha=0.3)

# B – Linear Regression: actual vs predicted
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor('#FFFFFF')
ax2.plot(test_df['date'], y_test.values,  color=BLUE, lw=2, label='Actual')
ax2.plot(test_df['date'], y_pred_lr,      color=PINK, lw=2, linestyle='--', label='LR Predicted')
ax2.set_title('B  Linear Regression — Test Set', fontweight='bold')
ax2.set_ylabel('Sales (₹)')
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'₹{x/1000:.0f}K'))
ax2.legend(fontsize=9); ax2.grid(axis='y', alpha=0.3)
ax2.text(0.05, 0.9, f'R² = {r2_lr:.3f}', transform=ax2.transAxes,
         fontsize=11, color=PINK, fontweight='bold')

# C – Polynomial Regression + 12-mo forecast
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor('#FFFFFF')
ax3.plot(test_df['date'], y_test.values,  color=BLUE,   lw=2, label='Actual')
ax3.plot(test_df['date'], y_pred_poly,    color=GREEN,  lw=2, linestyle='--', label='Poly Predicted')
ax3.plot(future_dates,    forecast,       color=PURPLE, lw=2.5, linestyle=':', label='2025 Forecast')
ax3.fill_between(future_dates, forecast * 0.93, forecast * 1.07,
                 alpha=0.15, color=PURPLE, label='±7% CI')
ax3.set_title('C  Polynomial Regression + 12-Month Forecast', fontweight='bold')
ax3.set_ylabel('Sales (₹)')
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'₹{x/1000:.0f}K'))
ax3.legend(fontsize=8); ax3.grid(axis='y', alpha=0.3)
ax3.text(0.05, 0.9, f'R² = {r2_poly:.3f}', transform=ax3.transAxes,
         fontsize=11, color=GREEN, fontweight='bold')

# Bottom comparison table
cmp = (f"  {'Model':<25} {'MAE':>8}   {'RMSE':>8}   {'R²':>6}  \n"
       f"  {'─'*55}\n"
       f"  {'Linear Regression':<25} ₹{mae_lr/1000:>5.1f}K   ₹{rmse_lr/1000:>5.1f}K   {r2_lr:>6.4f}  \n"
       f"  {'Polynomial (deg=3)':<25} ₹{mae_poly/1000:>5.1f}K   ₹{rmse_poly/1000:>5.1f}K   {r2_poly:>6.4f}  ")
fig.text(0.5, -0.02, cmp, ha='center', fontsize=9, fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='white', edgecolor='#CCCCCC', alpha=0.9))

plt.suptitle('Predictive Analytics — Historical Sales Forecasting',
             fontsize=16, fontweight='bold')

out_img = '/mnt/user-data/outputs/predictive_analytics_report.png'
plt.savefig(out_img, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n✓ Chart saved → {out_img}")

# Save forecast CSV
forecast_df = pd.DataFrame({'date': future_dates, 'forecasted_sales': forecast.round(2)})
csv_out = '/mnt/user-data/outputs/sales_forecast_2025.csv'
forecast_df.to_csv(csv_out, index=False)
print(f"✓ Forecast CSV → {csv_out}")
print("\n── Done! ──────────────────────────────────")
