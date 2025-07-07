import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#参数设定
csv_path = r'D:\Desktop\SURF-YOLFin\#DATA\#data_BTC_daily.csv'
output_base_folder = r'D:\Desktop\SURF-YOLFin\Bar\Bar_Charting\train'
window_size = 7
reference_days = 4
benchmark_days = 15
factor_name = 'OC_HL'

#创建目录
for label in ['buy', 'hold', 'sell']:
    os.makedirs(os.path.join(output_base_folder, label), exist_ok=True)

#处理
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

#构造
df[factor_name] = (df['Open'] - df['Close']) / (df['High'] - df['Low'])
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
df = df.reset_index(drop=True)

#计算 ARR
AR_list = []
for i in range(len(df) - window_size - reference_days + 1):
    future = df.iloc[i + window_size : i + window_size + reference_days]
    ret = future['Close'].pct_change().dropna() * 100
    AR = ret.mean()
    AR_list.append(AR)

AR_arr = np.array(AR_list)
p33 = np.percentile(AR_arr, 33)
p66 = np.percentile(AR_arr, 66)


count = 0
for i in range(len(df) - window_size - benchmark_days + 1):
    window_df = df.iloc[i : i + window_size]
    benchmark_df = df.iloc[i + window_size : i + window_size + benchmark_days]

    future_ret = benchmark_df['Close'].pct_change().dropna() * 100
    ABR = future_ret.mean()


    if ABR >= p66:
        label = 'buy'
    elif ABR <= p33:
        label = 'sell'
    else:
        label = 'hold'

    #标准化
    values = window_df[factor_name].values
    if np.any(np.isnan(values)) or np.any(np.isinf(values)):
        continue
    values = (values - np.mean(values)) / np.std(values)

    #绘图
    plt.figure(figsize=(2.24, 2.24), dpi=100)
    colors = ['red' if v < 0 else 'green' for v in values]
    plt.bar(range(window_size), values, color=colors)
    plt.ylim(-3, 3)
    plt.axis('off')
    plt.tight_layout(pad=0)


    filename = f'{count:06d}_{label}_B.png'
    save_path = os.path.join(output_base_folder, label, filename)
    plt.savefig(save_path)
    plt.close()
    count += 1

print(f'✅ 图像生成完毕，共生成 {count} 张，路径：{output_base_folder}')
