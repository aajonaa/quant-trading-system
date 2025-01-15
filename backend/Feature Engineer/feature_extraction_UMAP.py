import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
import numpy as np
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# 设置中文字体和负号显示
mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

# 定义函数选择特征
def select_features(X_scaled, umap_result, feature_names):
    correlation_matrix = np.corrcoef(np.hstack([X_scaled, umap_result]).T)
    corr_umap_dim1 = correlation_matrix[:len(feature_names), -3]
    corr_umap_dim2 = correlation_matrix[:len(feature_names), -2]
    corr_umap_dim3 = correlation_matrix[:len(feature_names), -1]

    # 选择相关性绝对值大于 0.2 的特征
    selected_features_dim1 = [feature_names[i] for i in np.where(np.abs(corr_umap_dim1) > 0.2)[0]]
    selected_features_dim2 = [feature_names[i] for i in np.where(np.abs(corr_umap_dim2) > 0.2)[0]]
    selected_features_dim3 = [feature_names[i] for i in np.where(np.abs(corr_umap_dim3) > 0.2)[0]]
    all_selected_features = set(selected_features_dim1).union(set(selected_features_dim2)).union(set(selected_features_dim3))
    return all_selected_features

# 定义函数绘制原始特征与降维后维度的相关性热力图
def plot_correlation_heatmap(X_scaled, umap_result, feature_names):
    plt.figure(figsize=(22,12))
    correlation_matrix = np.corrcoef(np.hstack([X_scaled, umap_result]).T)
    corr_umap_dim1 = correlation_matrix[:len(feature_names), -3]
    corr_umap_dim2 = correlation_matrix[:len(feature_names), -2]
    corr_umap_dim3 = correlation_matrix[:len(feature_names), -1]
    
    corr_df = pd.DataFrame({
        '特征': feature_names,
        '与 UMAP 维度1的相关性': corr_umap_dim1,
        '与 UMAP 维度2的相关性': corr_umap_dim2,
        '与 UMAP 维度3的相关性': corr_umap_dim3
    })
    corr_df = corr_df.set_index('特征')
    sns.heatmap(corr_df, annot=True, cmap='coolwarm')
    plt.title('原始特征与 UMAP 维度的相关性热力图')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 定义函数进行 UMAP 降维并可视化
def perform_umap(data):
    # 分离特征和目标变量
    feature_names = data.drop(['日期', '即期汇率'], axis=1).columns
    X = data.drop(['日期', '即期汇率'], axis=1)
    y = data['即期汇率']

    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 执行 UMAP 降维 (3维)
    umap_model = umap.UMAP(n_components=3, random_state=42)
    umap_result = umap_model.fit_transform(X_scaled)

    # 将 UMAP 结果与目标变量合并到一个新的 DataFrame 中
    umap_df = pd.DataFrame(data=umap_result, columns=['UMAP 维度1', 'UMAP 维度2', 'UMAP 维度3'])
    umap_df['即期汇率'] = y

    # 创建3D散点图
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(
        umap_df['UMAP 维度1'],
        umap_df['UMAP 维度2'],
        umap_df['UMAP 维度3'],
        c=umap_df['即期汇率'],
        cmap='viridis',
        alpha=0.6
    )
    
    ax.set_xlabel('UMAP 维度1')
    ax.set_ylabel('UMAP 维度2')
    ax.set_zlabel('UMAP 维度3')
    ax.set_title('UMAP 3D降维结果散点图')
    
    # 添加颜色条
    plt.colorbar(scatter, label='即期汇率')
    
    # 设置最佳视角
    ax.view_init(elev=20, azim=45)
    plt.tight_layout()
    plt.show()

    # 调用函数绘制热力图
    plot_correlation_heatmap(X_scaled, umap_result, feature_names)

# 读取文件
df = pd.read_excel(r'/Users/suping/Desktop/特征工程/combined_data_processed.xlsx')
# 调用函数
perform_umap(df)
