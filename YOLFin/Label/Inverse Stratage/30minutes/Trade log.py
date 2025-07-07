import pandas as pd
import numpy as np
import os

# === 参数 ===
file_path = r"D:\Desktop\YOLFin\#DATA\Data_BTC_30m.csv"
initial_capital = 10000
tp_pct = 0.10
sl_pct = 0.06
deltaThresholdHigh = 0.48
deltaThresholdLow = -0.8

# === 数据加载与因子构造 ===
df = pd.read_csv(file_path)
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df['delta'] = (df['open'] - df['close']) / (df['high'] - df['low'] + 1e-9)
df['position'] = 0
df.loc[df['delta'] > deltaThresholdHigh, 'position'] = 1
df.loc[df['delta'] < deltaThresholdLow, 'position'] = -1
df['future_close'] = df['close'].shift(-1)

# === 收益计算 ===
long_return = (df['future_close'] - df['close']) / df['close']
short_return = (df['close'] - df['future_close']) / df['close']
df['return_pct'] = 0.0
df.loc[df['position'] == 1, 'return_pct'] = np.clip(long_return, -sl_pct, tp_pct)
df.loc[df['position'] == -1, 'return_pct'] = np.clip(short_return, -sl_pct, tp_pct)
df['strategy_equity'] = initial_capital * (1 + df['return_pct']).cumprod()

# === 构造两行式交易记录 ===
records = []
current_equity = initial_capital

for idx, row in df[df['position'] != 0].iterrows():
    direction = 'Long' if row['position'] == 1 else 'Short'
    entry_price = row['close']
    exit_time = row.name + pd.Timedelta(minutes=30)
    exit_price = df.loc[exit_time, 'close'] if exit_time in df.index else np.nan
    ret_pct = row['return_pct']
    pnl = ret_pct * current_equity
    current_equity *= (1 + ret_pct)

    # 开仓记录
    records.append({
        '类型': f"{'多头' if direction == 'Long' else '空头'}进场",
        '方向': direction,
        '时间': row.name,
        '价格': f"{entry_price:.2f} USDT",
        '盈亏金额': '',
        '盈亏比例': '',
        '账户权益': ''
    })

    # 平仓记录
    records.append({
        '类型': f"{'多头' if direction == 'Long' else '空头'}出场",
        '方向': direction,
        '时间': exit_time,
        '价格': f"{exit_price:.2f} USDT" if not np.isnan(exit_price) else '',
        '盈亏金额': f"{pnl:,.2f} USDT",
        '盈亏比例': f"{ret_pct * 100:.2f}%",
        '账户权益': f"{current_equity:,.2f} USDT"
    })

# === 导出交割单 ===
log_df = pd.DataFrame(records)
log_df = log_df[['类型', '方向', '时间', '价格', '盈亏金额', '盈亏比例', '账户权益']]
output_path = r"D:\Desktop\YOLFin\Label\Inverse Stratage\formatted_trade_log.csv"
log_df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"✅ 已生成交割记录：{output_path}")
