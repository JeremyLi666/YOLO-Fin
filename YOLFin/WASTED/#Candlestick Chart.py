import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
import os


csv_path = r'D:\Desktop\SURF-YOLFin\#DATA\#data_BTC_daily.csv'
output_base_folder = r'D:\Desktop\SURF-YOLFin\Inverse_factor_graph\train'
window_size = 5
drawdown_window = 15


for label in ['buy', 'hold', 'sell']:
    os.makedirs(os.path.join(output_base_folder, label), exist_ok=True)

# 数据读取与预处理
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()
df.rename(columns={
    'date': 'Date',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

#DrawdownStrength 
df['Max_High_Past15'] = df['High'].rolling(window=drawdown_window, min_periods=1).max()
df['DrawdownStrength'] = (df['Max_High_Past15'] - df['Close']) / df['Max_High_Past15']
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
df = df.reset_index(drop=True)


drawdown_vals = df['DrawdownStrength'].dropna().values
p33 = np.percentile(drawdown_vals, 33)
p66 = np.percentile(drawdown_vals, 66)


count = 0
for i in range(len(df) - window_size - drawdown_window + 1):
    window_df = df.iloc[i : i + window_size]
    tag_row = df.iloc[i + window_size - 1]

    val = tag_row['DrawdownStrength']
    if val >= p66:
        label = 'buy'
    elif val <= p33:
        label = 'sell'
    else:
        label = 'hold'


    window_data = window_df.copy()
    window_data.set_index('Date', inplace=True)

    fig, axlist = mpf.plot(
        window_data,
        type='candle',
        style='yahoo',
        volume=False,
        figratio=(1,1),
        returnfig=True,
        figsize=(2.24, 2.24),
        axisoff=True
    )
    filename = f'{count:06d}_{label}_C.png'
    fig.savefig(os.path.join(output_base_folder, label, filename), bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    count += 1

print(f'✅ C 图像生成完毕，共生成 {count} 张，路径：{output_base_folder}')
