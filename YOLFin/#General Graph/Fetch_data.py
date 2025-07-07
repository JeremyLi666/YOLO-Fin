import pandas as pd
import ccxt
import time
import os
from datetime import datetime, timedelta, timezone

# 显示设置
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 50000)

# 设置代理（如不使用请注释）
proxies = {'http': '127.0.0.1:7897', 'https': '127.0.0.1:7897'}

# 初始化交易所
exchange = ccxt.binance({
    'enableRateLimit': True,
    'proxies': proxies,
})

# 参数设置
symbol = 'BTC/USDT'
timeframe = '30m'
limit = 800

# 文件夹路径与文件路径
folder_path = r'D:\Desktop\YOLFin\#DATA'
output_file = os.path.join(folder_path, 'Data_BTC_30m.csv')
os.makedirs(folder_path, exist_ok=True)

# 指定起始时间：2018-01-01
current_time = datetime(2018, 1, 1, tzinfo=timezone.utc)

# 设定拉取终止时间
end_time = datetime(2025, 7, 1, tzinfo=timezone.utc)

# 是否写入 header（仅首次）
write_header = not os.path.exists(output_file)

# 主循环：滚动拉取
while current_time < end_time:
    since = int(current_time.timestamp() * 1000)
    print(f"\n拉取自：{current_time.strftime('%Y-%m-%d %H:%M')} UTC")

    try:
        ohlcv = exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, since=since, limit=limit)
    except Exception as e:
        print(f"  -> 报错: {str(e)}，等待5秒重试")
        time.sleep(5)
        continue

    if not ohlcv:
        print("  -> 无数据，跳过")
        current_time += timedelta(minutes=30 * limit)
        continue

    # 构造 DataFrame
    df = pd.DataFrame(ohlcv, columns=['MTS', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['MTS'], unit='ms') + timedelta(hours=8)  # 转为北京时间
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
    df['date'] = df['date'].dt.strftime('%Y-%m-%d %H:%M')

    # 写入文件
    df.to_csv(output_file, mode='a', header=write_header, index=False)
    write_header = False

    print(f"  -> 写入 {len(df)} 条，最后时间: {df['date'].iloc[-1]}")

    # 更新时间指针
    current_time = datetime.fromtimestamp(ohlcv[-1][0] / 1000, tz=timezone.utc) + timedelta(minutes=30)

    # 限频保护
    time.sleep(0.5)
