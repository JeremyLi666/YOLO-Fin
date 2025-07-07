import pandas as pd
import matplotlib.pyplot as plt
import os

# === 路径设置 ===
input_file = 'data_BTC_daily.csv'
output_folder_kline = './BTC_KLine_Segments'
output_folder_ma = './BTC_MA_Segments'
os.makedirs(output_folder_kline, exist_ok=True)
os.makedirs(output_folder_ma, exist_ok=True)

# === 数据读取与预处理 ===
df = pd.read_csv(input_file)
df.columns = df.columns.str.strip()  # 去掉列名空格（以防万一）

df.rename(columns={
    'date': 'Date',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

df['Date'] = pd.to_datetime(df['Date'])
df.reset_index(drop=True, inplace=True)

# === 全局 MA 计算 ===
df['MA5'] = df['Close'].rolling(window=5, min_periods=1).mean()
df['MA10'] = df['Close'].rolling(window=10, min_periods=1).mean()
df['MA20'] = df['Close'].rolling(window=20, min_periods=1).mean()

# === 按30天一段切割 ===
segments = []
start_idx = 0
while start_idx < len(df):
    end_idx = start_idx + 29  # 每段30天数据
    segment = df.iloc[start_idx:end_idx + 1]
    if not segment.empty:
        segments.append((start_idx, end_idx, segment))
    start_idx = end_idx + 1

# === 绘图循环 ===
for idx, (start_i, end_i, segment_df) in enumerate(segments):
    x_labels = list(range(1, len(segment_df) + 1))

    # === K线图 ===
    fig, ax = plt.subplots(figsize=(14, 6))
    for i in range(len(segment_df)):
        o = segment_df['Open'].iloc[i]
        h = segment_df['High'].iloc[i]
        l = segment_df['Low'].iloc[i]
        c = segment_df['Close'].iloc[i]
        color = 'red' if c >= o else 'green'
        ax.plot([x_labels[i], x_labels[i]], [l, h], color=color, linewidth=2)  # 高低线
        ax.plot([x_labels[i], x_labels[i]], [o, c], color=color, linewidth=8)  # 实体线加粗

    ax.set_ylabel('Price')
    ax.set_title(f'BTC K-Line: {segment_df["Date"].iloc[0].date()} ~ {segment_df["Date"].iloc[-1].date()}')
    ax.set_xticks(x_labels)
    ax.set_xticklabels(x_labels, rotation=90)
    ax.set_yticklabels([])  # 去掉Y轴数字
    plt.tight_layout()
    output_path_kline = os.path.join(
        output_folder_kline,
        f'BTC_KLine_{idx+1}_{segment_df["Date"].iloc[0].date()}_{segment_df["Date"].iloc[-1].date()}.png'
    )
    plt.savefig(output_path_kline, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # === MA图 ===
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(x_labels, segment_df['MA5'], label='MA5', color='blue', linewidth=1.8)
    ax.plot(x_labels, segment_df['MA10'], label='MA10', color='orange', linewidth=1.8)
    ax.plot(x_labels, segment_df['MA20'], label='MA20', color='purple', linewidth=1.8)
    ax.set_ylabel('Moving Average')
    ax.set_title(f'BTC MA: {segment_df["Date"].iloc[0].date()} ~ {segment_df["Date"].iloc[-1].date()}')
    ax.set_xticks(x_labels)
    ax.set_xticklabels(x_labels, rotation=90)
    ax.set_yticklabels([])
    plt.tight_layout()
    output_path_ma = os.path.join(
        output_folder_ma,
        f'BTC_MA_{idx+1}_{segment_df["Date"].iloc[0].date()}_{segment_df["Date"].iloc[-1].date()}.png'
    )
    plt.savefig(output_path_ma, dpi=300, bbox_inches='tight')
    plt.close(fig)

print(f"保存完成，总计生成 {len(segments)} 段, 每段2张图, 路径:\nK线图: {output_folder_kline}\nMA图: {output_folder_ma}")
