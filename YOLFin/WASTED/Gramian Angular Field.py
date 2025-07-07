import pandas as pd
import numpy as np
import os
from pyts.image import GramianAngularField
from PIL import Image

# === 参数 ===
window_size = 30
future_days = 15
image_size = 32
output_root = "./GAF_1"
label_names = ['buy', 'sell', 'hold']
label_dirs = {label: os.path.join(output_root, f'GAF_1_{label}') for label in label_names}

# === 创建文件夹 ===
for path in label_dirs.values():
    os.makedirs(path, exist_ok=True)

# === 数据读取 ===
csv_path = "your_file.csv"  # 替换为你的CSV路径
df = pd.read_csv(csv_path, parse_dates=["date"])
close = df['close'].values

# === 收益率函数 ===
def future_return(close, t, horizon=15):
    return np.mean((close[t+1:t+1+horizon] - close[t]) / close[t])

# === GAF生成 ===
def normalize_series(series):
    scaled = (series - np.min(series)) / (np.max(series) - np.min(series))
    return scaled * 2 - 1

def generate_gaf(series):
    gas = GramianAngularField(image_size=image_size, method='summation')
    return gas.fit_transform([normalize_series(series)])[0]

# === 收益率和标签预处理 ===
returns = []
for i in range(window_size, len(df) - future_days):
    r = future_return(close, i, future_days)
    returns.append((i, r))

# === 分位数划分 ===
rets = np.array([r for _, r in returns])
q60 = np.quantile(rets, 0.6)
q40 = np.quantile(rets, 0.4)

# === 图像生成并分类保存 ===
for i, ret in returns:
    if ret > q60:
        label = 'buy'
    elif ret < q40:
        label = 'sell'
    else:
        label = 'hold'

    window_data = close[i - window_size:i]
    gaf = generate_gaf(window_data)
    img = Image.fromarray(np.uint8((gaf + 1) / 2 * 255), mode='L')

    date_str = df['date'].iloc[i - 1].strftime('%Y%m%d')
    save_path = os.path.join(label_dirs[label], f"{date_str}_GAF1.png")
    img.save(save_path)
