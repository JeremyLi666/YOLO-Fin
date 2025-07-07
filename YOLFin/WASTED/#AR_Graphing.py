import pandas as pd
import numpy as np
import os
from pyts.image import GramianAngularField
from PIL import Image


csv_path = r'D:\Desktop\SURF-YOLFin\#DATA\#data_BTC_daily.csv'
output_root = r'D:\Desktop\SURF-YOLFin\Gramian\AR_Graph\train'


window_size = 32
future_days = 15
image_size = 32         
save_image_size = 224   


label_dirs = {
    'buy': os.path.join(output_root, 'buy'),
    'sell': os.path.join(output_root, 'sell'),
    'hold': os.path.join(output_root, 'hold'),
}
for path in label_dirs.values():
    os.makedirs(path, exist_ok=True)

df = pd.read_csv(csv_path, parse_dates=['date'])
close = df['close'].values


def normalize(series):
    scaled = (series - np.min(series)) / (np.max(series) - np.min(series))
    return scaled * 2 - 1

def generate_gaf(series):
    transformer = GramianAngularField(image_size=image_size, method='summation')
    return transformer.fit_transform([normalize(series)])[0]

def future_return(close, t, horizon=15):
    return np.mean((close[t+1:t+1+horizon] - close[t]) / close[t])

#标签生成
returns = []
for i in range(window_size, len(close) - future_days):
    r = future_return(close, i, future_days)
    returns.append((i, r))

rets = np.array([r for _, r in returns])
q80 = np.quantile(rets, 0.8)
q20 = np.quantile(rets, 0.2)

#GAF图像生成并保存
for idx, (i, ret) in enumerate(returns):
    if ret > q80:
        label = 'buy'
    elif ret < q20:
        label = 'sell'
    else:
        label = 'hold'

    window = close[i - window_size:i]
    gaf = generate_gaf(window)
    gray = np.uint8((gaf + 1) / 2 * 255)
    rgb_array = np.stack([gray]*3, axis=-1)
    img = Image.fromarray(rgb_array, mode='RGB')
    img_resized = img.resize((save_image_size, save_image_size), resample=Image.BICUBIC)

    filename = f"{idx}_{label}_G.png"
    img_resized.save(os.path.join(label_dirs[label], filename), quality=95)
