import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === 路径 ===
file_path = r"D:\Desktop\YOLFin\#DATA\Data_BTC_m.csv"
epsilon = 1e-6

# === 数据读取 ===
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

# === 计算 delta 因子 ===
df['delta'] = (df['open'] - df['close']) / (df['high'] - df['low'] + epsilon)

# === 可视化分布 ===
plt.figure(figsize=(14, 6))
sns.histplot(df['delta'], bins=2000, kde=True, stat='density', color='darkorange')
plt.title(r"Distribution of $\delta = \frac{open - close}{high - low}$", fontsize=14)
plt.xlabel("delta", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

# 分位线标注
q5 = df['delta'].quantile(0.05)
q50 = df['delta'].quantile(0.5)
q95 = df['delta'].quantile(0.95)

plt.axvline(q5, color='red', linestyle='--', label=f'5% = {q5:.4f}')
plt.axvline(q50, color='black', linestyle='--', label=f'median = {q50:.4f}')
plt.axvline(q95, color='green', linestyle='--', label=f'95% = {q95:.4f}')
plt.legend()

plt.tight_layout()
plt.show()