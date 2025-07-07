import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === 读取原始数据 ===
file_path = r"D:\Desktop\YOLFin\#DATA\Data_BTC_m.csv"
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

#参数设置
input_window = 45      # 图像窗口长度
future_window = 15     # 预测收益窗口长度（分钟）

# === 计算未来15分钟收益率：r[t, t+15]
df['future_return'] = df['close'].shift(-future_window) / df['close'] - 1

#收益率绑定时间点为图像窗口起点：t - input_window
start_idx = input_window
end_idx = len(df) - future_window
returns = df.loc[start_idx:end_idx - 1, 'future_return']  #不做任何截断处理

#分位点
q5 = returns.quantile(0.05)
q50 = returns.quantile(0.5)
q95 = returns.quantile(0.95)

#可视化
plt.figure(figsize=(14, 6))
sns.histplot(returns, bins=2000, kde=True, stat='density', color='steelblue')
plt.title("Distribution of Future 15-min Returns (Post 45-min Image Window)", fontsize=14)
plt.xlabel("Future Return (r[t, t+15])", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

#添加分位线
plt.axvline(q5, color='red', linestyle='--', label=f'5% = {q5:.4f}')
plt.axvline(q50, color='black', linestyle='--', label=f'median = {q50:.4f}')
plt.axvline(q95, color='green', linestyle='--', label=f'95% = {q95:.4f}')
plt.legend()
plt.tight_layout()
plt.show()