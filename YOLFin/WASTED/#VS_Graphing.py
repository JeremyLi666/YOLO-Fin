import pandas as pd
import numpy as np
import os
from pyts.image import GramianAngularField
from PIL import Image

#参数设定
csv_path = r'D:\Desktop\SURF-YOLFin\#DATA\#data_BTC_daily.csv'
output_root = r'D:\Desktop\SURF-YOLFin\Gramian\VS_Graph\train'

window_size = 32      
image_size = 32        
save_image_size = 224  

#创建输出文件夹 
label_dirs = {
    'buy': os.path.join(output_root, 'buy'),
    'hold': os.path.join(output_root, 'hold'),
}
for path in label_dirs.values():
    os.makedirs(path, exist_ok=True)

# === 加载CSV数据 ===
df = pd.read_csv(csv_path, parse_dates=["date"])
close = df['close'].values

#工具函数
def normalize(series):
    scaled = (series - np.min(series)) / (np.max(series) - np.min(series))
    return scaled * 2 - 1

def generate_gaf(series, size=32):
    transformer = GramianAngularField(image_size=size, method='summation')
    series_norm = normalize(series)
    return transformer.fit_transform([series_norm])[0]

#计算rolling标准差（
volatilities = []
for i in range(window_size, len(close)):
    vol = np.std(close[i - window_size:i])
    volatilities.append((i, vol))

# === 分位数划分 ===
vol_vals = np.array([v for _, v in volatilities])
q40 = np.quantile(vol_vals, 0.4)

#图像生成 
for idx, (i, vol) in enumerate(volatilities):
    if vol < q40:
        label = 'buy'
    else:
        label = 'hold'

    window = close[i - window_size:i]
    gaf = generate_gaf(window, size=image_size)
    gray = np.uint8((gaf + 1) / 2 * 255)
    rgb_array = np.stack([gray] * 3, axis=-1)
    img = Image.fromarray(rgb_array, mode='RGB')
    img_resized = img.resize((save_image_size, save_image_size), resample=Image.BICUBIC)

    filename = f"{idx}_{label}_G.png"
    img_resized.save(os.path.join(label_dirs[label], filename), quality=95)

print("Volatility Shift 因子图像已生成完成，存储于 'VS_Graph/train/' 结构下。")
