import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import seaborn as sns

# 设置中文字体和负号显示
mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

def perform_pca(data, n_components=None):
    # 分离特征和目标变量
    X = data.drop(['日期', '即期汇率'], axis=1)
    y = data['即期汇率']
    feature_names = X.columns

    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 执行PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X_scaled)

    # 计算并打印每个主成分解释的方差百分比
    explained_variance_ratio = pca.explained_variance_ratio_
    print("每个主成分解释的方差百分比：")
    for i, ratio in enumerate(explained_variance_ratio, 1):
        print(f"主成分 {i}: {ratio * 100:.2f}%")

    # 累计方差贡献率
    cumulative_variance_ratio = explained_variance_ratio.cumsum()
    print("\n累计方差贡献率：")
    for i, ratio in enumerate(cumulative_variance_ratio, 1):
        print(f"前 {i} 个主成分: {ratio * 100:.2f}%")

    # 将主成分与目标变量合并到一个新的DataFrame中
    columns = [f'主成分{i+1}' for i in range(principal_components.shape[1])]
    principal_components_df = pd.DataFrame(data=principal_components, columns=columns)
    principal_components_df['即期汇率'] = y
    
    # 保存主成分分析结果
    save_pca_results(principal_components_df, pca, feature_names, X)

    return principal_components_df, pca

def save_pca_results(principal_components_df, pca, feature_names, X, output_dir='pca_results'):
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存主成分得分
    principal_components_df.to_csv(f'{output_dir}/pca_scores.csv', index=False)
    
    # 保存特征载荷
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(pca.n_components_)],
        index=feature_names
    )
    loadings.to_csv(f'{output_dir}/pca_loadings.csv')
    
    # 保存解释方差比率
    explained_variance_df = pd.DataFrame({
        '主成分': [f'PC{i+1}' for i in range(pca.n_components_)],
        '解释方差比率': pca.explained_variance_ratio_,
        '累计解释方差比率': pca.explained_variance_ratio_.cumsum()
    })
    explained_variance_df.to_csv(f'{output_dir}/explained_variance.csv', index=False)

def plot_pca(principal_components_df, pca):
    # 创建一个图形，包含多个子图
    fig = plt.figure(figsize=(15, 10))
    
    # 1. 前两个主成分的散点图
    plt.subplot(221)
    plt.scatter(principal_components_df['主成分1'], 
                principal_components_df['主成分2'], 
                c=principal_components_df['即期汇率'], 
                cmap='viridis', 
                alpha=0.5)
    plt.xlabel('主成分1')
    plt.ylabel('主成分2')
    plt.title('前两个主成分的散点图')
    plt.colorbar(label='即期汇率')
    
    # 2. 解释方差比率图
    plt.subplot(222)
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
             pca.explained_variance_ratio_, 'bo-')
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
             pca.explained_variance_ratio_.cumsum(), 'ro-')
    plt.xlabel('主成分数量')
    plt.ylabel('解释方差比率')
    plt.title('解释方差比率和累积解释方差比率')
    plt.legend(['单个解释方差比率', '累积解释方差比率'])
    plt.grid(True)
    
    # 3. 如果有第三个主成分，绘制3D散点图
    if principal_components_df.shape[1] >= 4:  # 包括即期汇率列
        ax = fig.add_subplot(223, projection='3d')
        scatter = ax.scatter(principal_components_df['主成分1'],
                           principal_components_df['主成分2'],
                           principal_components_df['主成分3'],
                           c=principal_components_df['即期汇率'],
                           cmap='viridis')
        ax.set_xlabel('主成分1')
        ax.set_ylabel('主成分2')
        ax.set_zlabel('主成分3')
        ax.set_title('前三个主成分的3D散点图')
        plt.colorbar(scatter)
    
    plt.tight_layout()
    plt.show()

def extract_significant_features(loadings_df, threshold=0.2):
    """
    从主成分载荷中提取显著特征
    """
    significant_features = {}
    for pc in loadings_df.columns:
        # 获取绝对值大于阈值的特征
        significant = loadings_df[abs(loadings_df[pc]) > threshold][pc]
        significant_features[pc] = significant.sort_values(ascending=False)
    
    return significant_features

# 读取文件
df = pd.read_excel(r'/Users/suping/Desktop/特征工程/combined_data_processed.xlsx')

# 调用函数进行主成分分析（保留95%的方差）
principal_components_df, pca = perform_pca(df, n_components=0.95)

# 调用函数进行可视化
plot_pca(principal_components_df, pca)

# 获取特征名称和载荷
feature_names = df.drop(['日期', '即期汇率'], axis=1).columns
loadings_df = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(pca.n_components_)],
    index=feature_names
)

# 提取显著特征
significant_features = extract_significant_features(loadings_df, threshold=0.2)

# 打印显著特征
print("\n显著特征(|载荷| > 0.2)：")
for pc, features in significant_features.items():
    print(f"\n{pc}的显著特征：")
    print(features)

