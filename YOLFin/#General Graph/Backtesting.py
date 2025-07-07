import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os

# === 设置文件路径（修改为你的路径）
file_path = r"D:\Desktop\YOLFin\#DATA\Data_BTC_30m.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"未找到文件：{file_path}")

# === 参数设置 ===
initial_capital = 10000
tp_pct = 0.10  # 止盈10%
sl_pct = 0.06  # 止损6%
deltaThresholdHigh = 0.48
deltaThresholdLow = -0.8

# === 读取数据 ===
df = pd.read_csv(file_path)
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# === 计算因子 delta ===
df['delta'] = (df['open'] - df['close']) / (df['high'] - df['low'] + 1e-9)

# === 生成交易信号 ===
df['position'] = 0
df.loc[df['delta'] > deltaThresholdHigh, 'position'] = 1
df.loc[df['delta'] < deltaThresholdLow, 'position'] = -1

# === 计算每笔交易收益率（百分比） ===
df['future_close'] = df['close'].shift(-1)
long_return = (df['future_close'] - df['close']) / df['close']
short_return = (df['close'] - df['future_close']) / df['close']

df['return_pct'] = 0.0
df.loc[df['position'] == 1, 'return_pct'] = np.clip(long_return, -sl_pct, tp_pct)
df.loc[df['position'] == -1, 'return_pct'] = np.clip(short_return, -sl_pct, tp_pct)

# === 构建净值曲线（策略 & 持有）===
df['strategy_equity'] = initial_capital * (1 + df['return_pct']).cumprod()
df['btc_equity'] = initial_capital * (df['close'] / df['close'].iloc[0])

# === 年度统计 ===
df['year'] = df.index.year
yearly_stats = []

for year, group in df.groupby('year'):
    start_value = group['strategy_equity'].iloc[0]
    end_value = group['strategy_equity'].iloc[-1]
    annual_return = (end_value / start_value - 1) * 100  # 百分比

    trades = group[group['position'] != 0]
    long_trades = trades[trades['position'] == 1]
    short_trades = trades[trades['position'] == -1]

    total_trades = len(trades)
    equity_curve = group['strategy_equity']
    roll_max = equity_curve.cummax()
    drawdown = equity_curve / roll_max - 1
    max_dd = drawdown.min() * 100

    sharpe = group['return_pct'].mean() / group['return_pct'].std() * np.sqrt(365 * 48) if group['return_pct'].std() != 0 else np.nan

    avg_long_gain = long_trades[long_return > 0]['return_pct'].mean() * 100 if not long_trades.empty else 0
    avg_long_loss = long_trades[long_return <= 0]['return_pct'].mean() * 100 if not long_trades.empty else 0
    avg_short_gain = short_trades[short_return > 0]['return_pct'].mean() * 100 if not short_trades.empty else 0
    avg_short_loss = short_trades[short_return <= 0]['return_pct'].mean() * 100 if not short_trades.empty else 0

    win_rate = (trades['return_pct'] > 0).mean() * 100 if not trades.empty else 0

    yearly_stats.append([
        year, round(annual_return, 2), total_trades, round(max_dd, 2),
        len(long_trades), len(short_trades),
        round(avg_long_gain, 2), round(avg_long_loss, 2),
        round(avg_short_gain, 2), round(avg_short_loss, 2),
        round(win_rate, 2), round(sharpe, 2)
    ])

columns = [
    'Year', 'Annual Return (%)', 'Total Trades', 'Max Drawdown (%)',
    'Long Trades', 'Short Trades',
    'Avg Long Gain (%)', 'Avg Long Loss (%)',
    'Avg Short Gain (%)', 'Avg Short Loss (%)',
    'Win Rate (%)', 'Sharpe Ratio'
]

stats_df = pd.DataFrame(yearly_stats, columns=columns)
stats_df.set_index('Year', inplace=True)

# === 绘图 ===
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(df.index, df['strategy_equity'], label='Strategy Equity')
ax.plot(df.index, df['btc_equity'], label='BTC Hold Equity', linestyle='--')
ax.set_ylabel('Portfolio Value (k USD)')
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x/1000:.1f}k'))
ax.set_title('Strategy vs BTC Buy & Hold with Annual Statistics')
ax.legend()
ax.grid(True)

# === 表格嵌入图中（字体16号，放大表格） ===
table_data = stats_df.astype(str).values
table = plt.table(
    cellText=table_data,
    colLabels=stats_df.columns,
    rowLabels=stats_df.index,
    cellLoc='center',
    loc='bottom',
    bbox=[0.0, -0.7, 1.0, 0.6]
)

table.set_fontsize(16)
table.scale(1, 2.0)  # 放大表格行高和列宽

plt.subplots_adjust(left=0.05, right=0.95, bottom=0.35)
plt.show()
