import pandas as pd
import numpy as np
import os

# === 路径设定 ===
input_path = r"D:\Desktop\YOLFin\#DATA\Data_BTC_30m.csv"
output_dir = r"D:\Desktop\YOLFin\Label\Inverse Stratage\30分钟级"
output_filename = "delta_labels_30min_reverse.csv"
output_path = os.path.join(output_dir, output_filename)

# === 参数设定 ===
epsilon = 1e-6
delta_buy_threshold = 0.48    # 高位买入（逆势）
delta_sell_threshold = -0.8   # 低位卖出（逆势）

# === 创建输出目录（若不存在）===
os.makedirs(output_dir, exist_ok=True)

# === 数据读取与处理 ===
df = pd.read_csv(input_path)
df.columns = df.columns.str.strip()
df['date'] = pd.to_datetime(df['date'])

# === delta因子计算 ===
df['delta'] = (df['open'] - df['close']) / (df['high'] - df['low'] + epsilon)

# === 标签生成函数 ===
def assign_label(delta):
    if delta > delta_buy_threshold:
        return 1
    elif delta < delta_sell_threshold:
        return -1
    else:
        return 0

df['label'] = df['delta'].apply(assign_label)

# === 保存输出文件 ===
df[['date', 'label']].to_csv(output_path, index=False)
print(f"标签已保存至：{output_path}")
