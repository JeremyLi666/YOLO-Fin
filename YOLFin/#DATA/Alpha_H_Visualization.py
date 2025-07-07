import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === 参数设置 ===
file_path = r"D:\Desktop\YOLFin\#DATA\#data_BTC_1h.csv"
epsilon = 1e-6  # 防止分母为零

# === 数据读取 ===
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

# === 计算 delta ===
df['delta'] = (df['open'] - df['close']) / (df['high'] - df['low'] + epsilon)

# === 去极值处理（防止尾部干扰）===
df = df[df['delta'].between(-10, 10)]

# === 可视化 ===
plt.figure(figsize=(12, 6))
sns.histplot(df['delta'], bins=300, kde=True, stat='density', color='darkorange', edgecolor=None)
plt.title(r"Distribution of $\delta = \frac{open - close}{high - low}$", fontsize=16)
plt.xlabel("delta", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
