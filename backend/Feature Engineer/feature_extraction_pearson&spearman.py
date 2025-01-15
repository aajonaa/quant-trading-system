from turtledemo.penrose import start

import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import mpl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 设置中文字体和负号显示
mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

def calculate_pearson_corr(df):
    """
    计算特征与即期汇率的皮尔逊相关系数

    """
    pearson_corr = {}
    for column in df.columns:
        if column != '即期汇率':
            # 计算皮尔逊相关系数和 p 值
            corr, _ = pearsonr(df[column], df['即期汇率'])
            if corr != 1:
                pearson_corr[column] = corr
    pearson_corr_df = pd.DataFrame(list(pearson_corr.items()), columns=['特征', '皮尔逊相关系数'])
    return pearson_corr_df[pearson_corr_df['皮尔逊相关系数'].abs() > 0.6]



def calculate_spearman_corr(df):
    """
    计算特征与即期汇率的斯皮尔曼相关系数
    参数:df -- 输入的数据框
    返回:spearman_corr_df -- 包含斯皮尔曼相关系数的数据框
    """
    spearman_corr = {}
    for column in df.columns:
        if column != '即期汇率':
            # 计算斯皮尔曼相关系数和 p 值
            corr, _ = spearmanr(df[column], df['即期汇率'])
            if corr != 1:
                spearman_corr[column] = corr
    spearman_corr_df = pd.DataFrame(list(spearman_corr.items()), columns=['特征', '斯皮尔曼相关系数'])
    return spearman_corr_df[spearman_corr_df['斯皮尔曼相关系数'].abs() > 0.6]


def plot_correlation(corr_df, corr_type):
    """
    绘制相关系数的柱状图
    参数:
    corr_df -- 包含相关系数的数据框
    corr_type -- 相关系数类型（如'皮尔逊'或'斯皮尔曼'）
    """
    plt.figure(figsize=(14,10))
    ax = sns.barplot(x='特征', y=f'{corr_type}相关系数', data=corr_df, palette='viridis',width=0.5)

    plt.title(f'{corr_type}相关系数与即期汇率')
    plt.ylabel(f'{corr_type}相关系数')
    plt.xticks(rotation=45, ha='right')

    # 添加数值标签
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.2f'),
                    (p.get_x() + p.get_width() / 2., p.get_height()+0.02),
                    ha='center', va='bottom',
                    xytext=(0, 10),color='black',
                    textcoords='data')

    plt.tight_layout()
    plt.show()

# 读取文件
df = pd.read_excel(r'C:\Users\80651\Documents\WeChat Files\wxid_i96gfzaamhyn22\FileStorage\File\2025-02\特征工程\combined_data_processed.xlsx')
# 排除日期列
df = df.drop(columns=['日期'])
# 计算皮尔逊相关系数
pearson_corr_df = calculate_pearson_corr(df)
# 绘制皮尔逊相关系数柱状图
plot_correlation(pearson_corr_df, '皮尔逊')
# 计算斯皮尔曼相关系数
spearman_corr_df = calculate_spearman_corr(df)
# 绘制斯皮尔曼相关系数柱状图
plot_correlation(spearman_corr_df, '斯皮尔曼')