from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
import pandas as pd
from pylab import mpl

# 设置中文字体和负号显示
mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False


def load_data(file_path):
    """
    加载数据并去除值全为 0 的行
    :param file_path: 文件路径
    :return: 处理后的数据
    """
    data = pd.read_excel(file_path)
    data = data[(data != 0).all(axis=1)]
    return data


# 定义函数来进行 STL 分解
def stl_decomposition(data, period):
    # 进行 STL 分解
    stl = STL(data, period=period)
    res = stl.fit()
    return res.seasonal


def combine_data(growth_rate_diff_seasonal, exchange_rate_seasonal):
    """
    合并季节性数据
    :param growth_rate_diff_seasonal: 增长率差值的季节性数据
    :param exchange_rate_seasonal: 即期汇率的季节性数据
    """
    # 合并结果为数据框
    combined_df = pd.DataFrame({
        '中国和美国GDP增长率差值的季节性数据': growth_rate_diff_seasonal,
        '人民币兑换美元即期汇率的季节性数据': exchange_rate_seasonal
    })

    # 使用线性插值填充缺失值
    combined_df = combined_df.interpolate(method='linear')

    return combined_df


def plot_data(combined_df):
    """
    对合并后的数据进行可视化
    :param combined_df: 合并后的数据框
    """
    # 绘制合并后的数据框的可视化图形
    combined_df.plot(figsize=(12, 6),
                     title='中国和美国GDP增长率差值与人民币兑换美元即期汇率的季节性数据对比')
    plt.xlabel('日期')
    plt.ylabel('季节性数值')
    plt.legend()
    plt.grid(True)
    plt.show()

def generate_exchange_rate_data4():
    # 中国季度 GDP 季度增长率
    china_gdp = load_data('/Users/suping/Desktop/特征工程/中国季度GDP.xlsx')
    china_gdp['日期'] = pd.to_datetime(china_gdp['日期'].str.replace('Q', '-'))
    china_gdp['中国GDP增长率'] = china_gdp['中国名义GDP(亿元)'].pct_change()

    # 美国季度 GDP 季度增长率
    us_gdp = load_data('/Users/suping/Desktop/特征工程/美国季度GDP.xlsx')
    us_gdp['日期'] = pd.to_datetime(us_gdp['日期'].str.replace('Q', '-'))
    us_gdp['美国GDP增长率'] = us_gdp['美国名义GDP(亿元)'].pct_change()

    # 合并中国和美国的 GDP 增长率数据
    merged_data = pd.merge(china_gdp[['日期', '中国GDP增长率']],
                           us_gdp[['日期', '美国GDP增长率']],
                           left_on='日期',
                           right_on='日期')
    # 计算中国和美国 GDP 增长率的差值
    merged_data['增长率差值'] = merged_data['中国GDP增长率'] - merged_data['美国GDP增长率']
    # 对增长率差值进行 STL 分解
    merged_data.set_index('日期', inplace=True)
    growth_rate_diff_seasonal = stl_decomposition(merged_data['增长率差值'].dropna(), 4)

    # 即期汇率
    exchange_rate = load_data('/Users/suping/Desktop/特征工程/人民币兑换美元日度数据.xlsx')
    # 对即期汇率进行 STL 分解
    exchange_rate.set_index('日期', inplace=True)
    exchange_rate_seasonal = stl_decomposition(exchange_rate['即期汇率'], 365)

    # 合并数据
    exchange_rate_new4 = combine_data(growth_rate_diff_seasonal, exchange_rate_seasonal)

    # 可视化数据
    plot_data(exchange_rate_new4)
    return exchange_rate_new4
if __name__ == "__main__":
    df = generate_exchange_rate_data4()
    print(df)
    print(df.columns)