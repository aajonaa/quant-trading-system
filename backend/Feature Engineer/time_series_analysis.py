#file1
import pandas as pd
import numpy as np
import ta
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
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


def calculate_moving_average(data, window):
    """
    计算移动平均线
    :param data: 数据
    :param window: 窗口大小
    :return: 移动平均线数据
    """
    ma = data['即期汇率'].rolling(window=window).mean()
    ma = ma.to_frame()
    ma = ma.rename(columns={'即期汇率': f'移动平均汇率({window})'})
    return ma


def calculate_std(data, window):
    """
    计算移动标准差
    :param data: 数据
    :param window: 窗口大小
    :return: 移动标准差数据
    """
    std = data['即期汇率'].rolling(window=window).std()
    return std


def calculate_rsi(data, window):
    """
    计算RSI指标
    :param data: 数据
    :param window: 窗口大小
    :return: RSI数据
    """
    rsi = ta.momentum.RSIIndicator(data['即期汇率'], window=window).rsi()
    return rsi


def calculate_volatility(data, window):
    """
    计算历史波动率（年化）
    :param data: 数据
    :param window: 窗口大小
    :return: 包含对数收益率和历史波动率的数据
    """
    log = np.log(data['即期汇率'] / data['即期汇率'].shift(1))
    vol = log.rolling(window=window).std() * np.sqrt(252)
    return vol


def calculate_trend(data):
    """
    计算趋势线和趋势斜率、趋势加速度
    :param data: 数据
    :return: 包含趋势线和趋势加速度的数据，趋势斜率
    """
    x = np.arange(len(data)).reshape(-1, 1)
    y = data['即期汇率'].values
    model = LinearRegression()
    model.fit(x, y)
    trend = model.predict(x)
    trend_slope = model.coef_[0]
    trend_acceleration = np.gradient(np.gradient(trend))
    return trend, trend_slope, trend_acceleration


def plot_bollinger_bands(data, window):
    """
    绘制布林带图
    :param data: 数据
    :param window: 窗口大小
    """
    plt.figure(figsize=(16, 8))
    plt.plot(data['日期'], data['即期汇率'], label='即期汇率', color='red',linewidth=1)
    plt.plot(data['日期'], data[f'移动平均汇率({window})'], label=f'中轨（{window}日移动平均汇率）', color='orange', linewidth=2)
    plt.plot(data['日期'], data['上轨'], label='上轨', color='green', linewidth=2)
    plt.plot(data['日期'], data['下轨'], label='下轨', color='green', linewidth=2)
    plt.title(f'2004年——2024年人民币兑换美元即期汇率布林带')
    plt.xlabel('日期')
    plt.xticks(rotation=45)
    plt.ylabel('汇率')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_rsi(data, window):
    """
    绘制RSI走势图
    :param data: 数据
    :param window: 窗口大小
    """
    plt.figure(figsize=(16, 8))
    # 修改列名与实际添加的列名一致
    plt.plot(data['日期'], data['RSI'], label=f'{window}日移动即期汇率的RSI')
    plt.title(f'2004年——2024年人民币兑换美元即期汇率{window}日移动即期汇率的RSI')
    plt.xlabel('日期')
    plt.xticks(rotation=45)
    plt.ylabel('RSI')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_volatility(data, window):
    """
    绘制历史波动率走势图
    :param data: 数据
    :param window: 窗口大小
    """
    plt.figure(figsize=(16, 8))
    # 修改列名与实际添加的列名一致
    plt.plot(data['日期'], data['历史波动率（年化）'], label=f'{window}日历史波动率（年化）', color='blue')
    plt.title(f'2004年——2024年人民币兑换美元即期汇率{window}日历史波动率（年化）')
    plt.xlabel('日期')
    plt.xticks(rotation=45)
    plt.ylabel('历史波动率')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_trend(data):
    """
    绘制趋势线图
    :param data: 数据
    """
    plt.figure(figsize=(16, 8))
    plt.plot(data['日期'], data['即期汇率'], label='即期汇率')
    plt.plot(data['日期'], data['趋势线'], label='趋势线', linestyle='--', color='orange')
    plt.title('2004年——2024年人民币兑换美元即期汇率趋势线')
    plt.xlabel('日期')
    plt.xticks(rotation=45)
    plt.ylabel('汇率')
    plt.legend()
    plt.grid(True)
    plt.show()


def generate_exchange_rate_data1():
    # 设定 window 的值
    window = 120
    # 加载数据
    file_path = '/Users/suping/Desktop/特征工程/人民币兑换美元日度数据.xlsx'
    exchange_rate = load_data(file_path)

    # 计算移动平均线
    exchange_rate_MA = calculate_moving_average(exchange_rate, window)
    exchange_rate_new1 = pd.concat([exchange_rate, exchange_rate_MA], axis=1)

    # 计算移动标准差
    exchange_rate_new1[f'移动标准差({window})'] = calculate_std(exchange_rate, window)

    # 计算布林带上下轨
    exchange_rate_new1['上轨'] = exchange_rate_new1[f'移动平均汇率({window})'] + 2 * exchange_rate_new1[f'移动标准差({window})']
    exchange_rate_new1['下轨'] = exchange_rate_new1[f'移动平均汇率({window})'] - 2 * exchange_rate_new1[f'移动标准差({window})']

    # 计算RSI
    exchange_rate_new1['RSI'] = calculate_rsi(exchange_rate, window)

    # 计算历史波动率
    exchange_rate_new1['历史波动率（年化）'] = calculate_volatility(exchange_rate, window)

    # 计算趋势线和趋势斜率、趋势加速度
    trend, trend_slope, trend_acceleration = calculate_trend(exchange_rate)
    exchange_rate_new1['趋势线'] = trend
    exchange_rate_new1['趋势加速度'] = trend_acceleration

    # 绘制布林带图
    plot_bollinger_bands(exchange_rate_new1, window)

    # 绘制RSI走势图
    plot_rsi(exchange_rate_new1, window)

    # 绘制历史波动率走势图
    plot_volatility(exchange_rate_new1, window)

    # 绘制趋势线图
    plot_trend(exchange_rate_new1)

    # 趋势斜率
    exchange_rate_new1.set_index('日期', inplace=True)
    return exchange_rate_new1


if __name__ == "__main__":
    df = generate_exchange_rate_data1()
    print(df)
