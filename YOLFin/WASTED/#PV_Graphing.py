import pandas as pd
import numpy as np
import os
from pyts.image import GramianAngularField
from PIL import Image

#参数设定
csv_path = r'D:\Desktop\SURF-YOLFin\#DATA\#data_BTC_daily.csv'
output_root = r'D:\Desktop\SURF-YOLFin\Gramian\PV_Graphing\train'

window_size = 32
image_size = 32
save_image_size = 224

#输出文件夹
label_dirs = {
    'buy': os.path.join(output_root, 'buy'),
    'sell': os.path.join(output_root, 'sell'),
    'hold': os.path.join(output_root, 'hold'),
}
for path in label_dirs.values():
    os.makedirs(path, exist_ok=True)

# 加载数据
df = pd.read_csv(csv_path, parse_dates=["date"])
close = df['close'].values

#函数定义
def normalize(series):
    scaled = (series - np.min(series)) / (np.max(series) - np.min(series))
    return scaled * 2 - 1

def generate_gaf(series, size=32):
    transformer = GramianAngularField(image_size=size, method='summation')
    series_norm = normalize(series)
    return transformer.fit_transform([series_norm])[0]

for idx in range(window_size, len(close)):
    window = close[idx - window_size:idx]
    current_price = close[idx - 1]
    min_price = np.min(window)
    max_price = np.max(window)
    relative_pos = (current_price - min_price) / (max_price - min_price + 1e-8)

    #分类标准
    if relative_pos >= 0.9:
        label = 'sell'
    elif relative_pos <= 0.1:
        label = 'buy'
    else:
        label = 'hold'

    #GAF图生成与保存
    gaf = generate_gaf(window)
    gray = np.uint8((gaf + 1) / 2 * 255)
    rgb_array = np.stack([gray] * 3, axis=-1)
    img = Image.fromarray(rgb_array, mode='RGB')
    img_resized = img.resize((save_image_size, save_image_size), resample=Image.BICUBIC)

    filename = f"{idx}_{label}_G.png"
    img_resized.save(os.path.join(label_dirs[label], filename), quality=95)

print("位置因子图像已成功生成，存储在 'PV_Graphing/train' 子文件夹中。")
