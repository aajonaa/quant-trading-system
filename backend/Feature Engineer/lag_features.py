import pandas as pd
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
#import seaborn as sns

def load_data(file_path):
    """
    加载数据并去除值全为0的行
    :param file_path: 文件路径
    :return: 处理后的数据
    """
    data = pd.read_excel(file_path)
    data = data[(data != 0).all(axis=1)]
    return data
def calculate_exchange_rate_changes(data, days_list):
    exchange_rate_changes_list = []
    for days in days_list:
        changes=data['即期汇率'].pct_change(periods=days)
        changes=changes.to_frame()
        changes=changes.rename(columns={'即期汇率':f'汇率变化率({days})'})
        exchange_rate_changes_list.append(changes)
    exchange_rate_changes = pd.concat(exchange_rate_changes_list, axis=1)
    return exchange_rate_changes
'''
def normalize_data(data):
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    normalized_df = pd.DataFrame(normalized_data, columns=data.columns)
    return normalized_df
def correlation_analysis(data, target_column='即期汇率', threshold=0.5):
    """
    进行相关性分析，选择与目标列相关性较高且相互之间冗余度较低的特征
    """
    # 筛选出数值型列
    numeric_data = data.select_dtypes(include='number')
    corr_matrix = numeric_data.corr()
    target_corr = corr_matrix[target_column].drop(target_column)
    high_corr_features = target_corr[abs(target_corr) > threshold].index

    # 去除高度相关的冗余特征
    final_features = []
    for feature in high_corr_features:
        is_redundant = False
        for selected_feature in final_features:
            if abs(corr_matrix.loc[feature, selected_feature]) > threshold:
                is_redundant = True
                break
        if not is_redundant:
            final_features.append(feature)
    return final_features, corr_matrix


def pca_analysis(data, n_components=None):
    """
    进行主成分分析
    """
    pca = PCA(n_components=n_components)
    # 在进行 PCA 前去除包含 NaN 值的行
    data = data.dropna()
    principal_components = pca.fit_transform(data)
    explained_variance_ratio = pca.explained_variance_ratio_
    return principal_components, explained_variance_ratio
'''
def generate_exchange_rate_data2():
    days_list = [1, 7, 30]
    file_path = '/Users/suping/Desktop/特征工程/人民币兑换美元日度数据.xlsx'
    exchange_rate = load_data(file_path)
    exchange_rate_changes = calculate_exchange_rate_changes(exchange_rate, days_list)
    exchange_rate_new2=pd.concat([exchange_rate, exchange_rate_changes], axis=1)
    exchange_rate_new2.set_index('日期', inplace=True)
    return exchange_rate_new2
'''
    columns_to_normalize = [col for col in exchange_rate_new2.columns if '汇率变化率' in col]
    normalized_part = normalize_data(exchange_rate_new2[columns_to_normalize])
    exchange_rate_new2[columns_to_normalize] = normalized_part

    # 相关性分析
    final_features, corr_matrix = correlation_analysis(exchange_rate_new2)
    print('相关性较高且冗余度较低的特征：', final_features)

    # 提取汇率变化率列用于 PCA
    pca_columns = [col for col in exchange_rate_new2.columns if '汇率变化率' in col]
    principal_components, explained_variance_ratio = pca_analysis(exchange_rate_new2[pca_columns])
    print('各主成分的方差解释比例：', explained_variance_ratio)

    # 可视化相关性热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('特征相关性热力图')
    plt.show()

    # 可视化主成分方差解释比例
    plt.figure(figsize=(8, 6))
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
    plt.xlabel('主成分')
    plt.ylabel('方差解释比例')
    plt.title('主成分方差解释比例')
    plt.show()
 '''
if __name__ == "__main__":
    df = generate_exchange_rate_data2()
    print(df)
