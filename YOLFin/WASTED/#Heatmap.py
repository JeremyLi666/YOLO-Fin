import pandas as pd
import numpy as np
import os
from PIL import Image

# === 参数设定 ===
csv_path = r'D:\Desktop\SURF-YOLFin\#DATA\#data_BTC_daily.csv'
output_root = r'D:\Desktop\SURF-YOLFin\Heat_HighRes\train'
window_size = 15
save_image_size = 224  # 最终图像尺寸

# === 分类文件夹创建 ===
label_dirs = {
    'buy': os.path.join(output_root, 'buy'),
    'sell': os.path.join(output_root, 'sell'),
    'hold': os.path.join(output_root, 'hold'),
}
for path in label_dirs.values():
    os.makedirs(path, exist_ok=True)

# === 加载数据并预处理 ===
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

# === 构造因子 ===
df['Accel'] = df['Close'] - 2 * df['Close'].shift(1) + df['Close'].shift(2)
df['Volatility'] = df['Close'].rolling(window=5).std()
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# === 阈值划分依据（分类标签） ===
p33 = np.percentile(df['Accel'], 33)
p66 = np.percentile(df['Accel'], 66)

# === 主循环：生成图像 ===
count = 0
for i in range(window_size, len(df)):
    window = df.iloc[i - window_size:i]

    # 提取三因子原始值
    acc = window['Accel'].values
    close = window['Close'].values
    vol = window['Volatility'].values

    # 分别归一化为 [-1, 1]
    def normalize(x):
        x = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)
        return x * 2 - 1

    r_norm = normalize(acc)
    g_norm = normalize(close)
    b_norm = normalize(vol)

    # 拓展为图像矩阵 (15, 1, 3) → (15, 32, 3)
    def expand_channel(c): return np.tile(c.reshape(-1, 1), (1, 32))
    rgb_matrix = np.stack([expand_channel(r_norm),
                           expand_channel(g_norm),
                           expand_channel(b_norm)], axis=2)  # shape: (15, 32, 3)

    # 转换为 uint8 图像并 resize 到 224×224
    rgb_uint8 = np.uint8((rgb_matrix + 1) / 2 * 255)  # [-1,1] → [0,255]
    img = Image.fromarray(rgb_uint8, mode='RGB')
    img_resized = img.resize((save_image_size, save_image_size), resample=Image.BICUBIC)

    # 标签打标：用最后一天 Accel 值
    final_alpha = acc[-1]
    if final_alpha <= p33:
        label = 'buy'
    elif final_alpha >= p66:
        label = 'sell'
    else:
        label = 'hold'

    # 保存图像
    filename = f"{i:06d}_{label}_H_RGB.png"
    img_resized.save(os.path.join(label_dirs[label], filename), quality=95)
    count += 1

print(f"✅ 已生成 {count} 张高质量 Heat 图像，保存在：{output_root}")
