import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import mpl
# 设置中文字体和负号显示
mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

def load_data(file_path):
    """
    加载数据并去除值全为0的行
    :param file_path: 文件路径
    :return: 处理后的数据
    """
    data = pd.read_excel(file_path)
    data = data[(data != 0).all(axis=1)]
    return data

# 定义函数计算不同时间窗口内的统计特征
def calculate_window_stats(data, date_column, value_column, window):
    stats = {}
    for window in window:
        rolled = data.set_index(date_column)[value_column].rolling(window)
        mean_values = rolled.mean()
        var_values = rolled.var()
        max_values = rolled.max()
        min_values = rolled.min()

        stats[window] = {
            'mean': mean_values,
            'var': var_values,
            'max': max_values,
           'min': min_values
        }

    return stats

def plot_time_window_stats(stats):
    for window, window_stats in stats.items():
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # 绘制均值曲线
        sns.lineplot(x=window_stats['mean'].index, y=window_stats['mean'], ax=axes[0, 0])
        axes[0, 0].set_title(f'{window}日窗口下均值变化')

        # 绘制方差曲线
        sns.lineplot(x=window_stats['var'].index, y=window_stats['var'], ax=axes[0, 1])
        axes[0, 1].set_title(f'{window}日窗口下方差变化')

        # 绘制最大值曲线
        sns.lineplot(x=window_stats['max'].index, y=window_stats['max'], ax=axes[1, 0])
        axes[1, 0].set_title(f'{window}日窗口下最大值变化')

        # 绘制最小值曲线
        sns.lineplot(x=window_stats['min'].index, y=window_stats['min'], ax=axes[1, 1])
        axes[1, 1].set_title(f'{window}日窗口下最小值变化')

        plt.tight_layout()
        plt.show()
        
def generate_exchange_rate_data3():
    window=[30,60,90]
    file_path = '/Users/suping/Desktop/特征工程/人民币兑换美元日度数据.xlsx'
    exchange_rate = load_data(file_path)
    stats = calculate_window_stats(exchange_rate, '日期', '即期汇率', window)
    plot_time_window_stats(stats)
    stats_df = []
    for window, window_stats in stats.items():
        window_df = pd.DataFrame({
            f'均值({window})': window_stats['mean'],
            f'方差({window})': window_stats['var'],
            f'最大值({window})': window_stats['max'],
            f'最小值({window})': window_stats['min']
        }).reset_index()
        stats_df.append(window_df)
    # 合并所有时间窗口的统计特征 DataFrame
    merged_stats_df = stats_df[0]
    for i in range(1, len(stats_df)):
        merged_stats_df = pd.merge(merged_stats_df, stats_df[i], on='日期', how='outer')
    # 与原始数据合并
    exchange_rate_new3= pd.merge(exchange_rate, merged_stats_df, on='日期', how='left')
    exchange_rate_new3.set_index('日期', inplace=True)
    return exchange_rate_new3
if __name__ == "__main__":
    df = generate_exchange_rate_data3()
    print(df)

