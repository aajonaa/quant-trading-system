import os
import numpy as np
import pandas as pd
from PyEMD import CEEMDAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging

class FeatureEngineer:
    def __init__(self):
        self.ceemdan = CEEMDAN()
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # 保留95%的方差
        self.logger = logging.getLogger(__name__)
        
    def process_data(self, df, pair):
        """主处理函数"""
        # 1. 生成特征
        df = self.generate_technical_features(df)
        df = self.process_macro_features(df, pair)
        df = self.generate_iceemdan_features(df)
        
        # 2. 保存原始特征到backend/FE目录（不做归一化）
        output_path = os.path.join(f'{pair}_processed.csv')
        df.to_csv(output_path, index=False)
        self.logger.info(f"已保存原始特征数据到 {output_path}")
        
        # 3. 数据预处理（处理缺失值、异常值等，但不做归一化）
        df_clean = self.preprocess_features(df)
        
        # 4. 对数据进行归一化，然后再进行PCA降维
        df_normalized = self.normalize_features(df_clean)
        
        # 5. 降维处理
        df_pca = self.dimension_reduction(df_normalized)
        pca_path = os.path.join(f'{pair}_PCA.csv')
        df_pca.to_csv(pca_path, index=False)
        self.logger.info(f"已保存PCA降维后的数据到 {pca_path}")
        
        return df_pca
    
    def generate_technical_features(self, df):
        """生成技术指标特征"""
        # 创建一个特征字典来存储所有特征
        features = {}
        
        # 1. 趋势特征
        for window in [5, 10, 20, 50]:
            # 移动平均
            features[f'MA{window}'] = df['Close'].rolling(window).mean()
            features[f'MA_dist_{window}'] = (df['Close'] - features[f'MA{window}']) / features[f'MA{window}']
            features[f'MA_slope_{window}'] = features[f'MA{window}'].diff(5) / features[f'MA{window}'].shift(5)
            
            # 价格动量
            features[f'momentum_{window}'] = df['Close'].diff(window)
            features[f'momentum_ma_{window}'] = features[f'momentum_{window}'].rolling(window//2).mean()
            
            # 波动率
            features[f'volatility_{window}'] = df['Close'].rolling(window).std()
            features[f'volatility_ratio_{window}'] = (
                features[f'volatility_{window}'] / 
                features[f'volatility_{window}'].rolling(window*2).mean()
            )
            
            # 价格范围
            features[f'price_range_{window}'] = (
                df['High'].rolling(window).max() - 
                df['Low'].rolling(window).min()
            ) / df['Close']
        
        # 2. 波动率特征
        features['returns'] = df['Close'].pct_change()
        features['volatility'] = features['returns'].rolling(20).std()
        
        # 高低价波动
        features['hl_volatility'] = (df['High'] - df['Low']) / df['Open']
        features['hl_volatility_ma'] = features['hl_volatility'].rolling(10).mean()
        
        # 日内波动
        features['intraday_intensity'] = (
            (df['Close'] - df['Open']) / (df['High'] - df['Low'])
        ).abs()
        
        # 3. 布林带增强
        for window in [10, 20, 50]:
            for std in [1.5, 2.0, 2.5]:
                features[f'BB_middle_{window}'] = df['Close'].rolling(window).mean()
                bb_std = df['Close'].rolling(window).std()
                features[f'BB_upper_{window}_{std}'] = features[f'BB_middle_{window}'] + std * bb_std
                features[f'BB_lower_{window}_{std}'] = features[f'BB_middle_{window}'] - std * bb_std
                features[f'BB_width_{window}_{std}'] = (
                    (features[f'BB_upper_{window}_{std}'] - features[f'BB_lower_{window}_{std}']) / 
                    features[f'BB_middle_{window}']
                )
        
        # 一次性合并所有特征
        feature_df = pd.concat(features, axis=1)
        
        # 合并原始数据和特征
        result = pd.concat([df, feature_df], axis=1)
        
        return result
    
    def process_macro_features(self, df, pair):
        """处理宏观经济指标"""
        # 确保df有Date列并且格式正确
        if 'Date' not in df.columns:
            df['Date'] = pd.to_datetime(df.index)
        else:
            df['Date'] = pd.to_datetime(df['Date'])
        
        # 创建一个字典来存储所有宏观特征
        macro_features = {}
        
        # 获取货币对对应的国家代码
        country_pairs = {
            'CNYAUD': ('CN', 'AU'),
            'CNYEUR': ('CN', 'EU'),
            'CNYGBP': ('CN', 'UK'),
            'CNYJPY': ('CN', 'JP'),
            'CNYUSD': ('CN', 'US')
        }
        
        if pair not in country_pairs:
            self.logger.error(f"未知的货币对: {pair}")
            return df
        
        country1, country2 = country_pairs[pair]
        
        # 需要处理的宏观指标
        indicators = ['CPI', 'INFLATION', 'REAL_GDP', 'UNEMPLOYMENT']
        
        for indicator in indicators:
            try:
                path1 = os.path.join('..', '..', 'try', 'macro_data', f'{country1}_{indicator}.csv')
                path2 = os.path.join('..', '..', 'try', 'macro_data', f'{country2}_{indicator}.csv')
                
                self.logger.info(f"读取宏观数据: {path1}")
                self.logger.info(f"读取宏观数据: {path2}")
                
                if not os.path.exists(path1) or not os.path.exists(path2):
                    self.logger.error(f"宏观数据文件不存在: {path1} 或 {path2}")
                    continue
                    
                df1 = pd.read_csv(path1)
                df2 = pd.read_csv(path2)
                
                # 重命名列
                df1 = df1.rename(columns={'date': 'Date'})
                df2 = df2.rename(columns={'date': 'Date'})
                
                # 确保日期格式正确
                df1['Date'] = pd.to_datetime(df1['Date'])
                df2['Date'] = pd.to_datetime(df2['Date'])
                
                # 设置日期索引并排序
                df1 = df1.set_index('Date').sort_index()
                df2 = df2.set_index('Date').sort_index()
                
                # 对齐日期范围
                start_date = max(df1.index.min(), df2.index.min(), df['Date'].min())
                end_date = min(df1.index.max(), df2.index.max(), df['Date'].max())
                
                # 重采样并插值
                df1 = df1.resample('D').interpolate(method='linear')
                df2 = df2.resample('D').interpolate(method='linear')
                
                # 截取共同的日期范围
                df1 = df1[start_date:end_date]
                df2 = df2[start_date:end_date]
                
                # 计算差值
                macro_features[f'{indicator}_{indicator}_diff'] = df1[indicator] - df2[indicator]
                
                # 计算同比差值
                yoy_col = f'{indicator}_YOY'
                if yoy_col in df1.columns and yoy_col in df2.columns:
                    macro_features[f'{indicator}_{indicator}_YOY_diff'] = df1[yoy_col] - df2[yoy_col]
                    
            except Exception as e:
                self.logger.error(f"处理{indicator}时出错: {str(e)}")
                self.logger.error("错误详情: ", exc_info=True)
                continue
        
        # 一次性合并所有宏观特征
        macro_df = pd.concat(macro_features, axis=1)
        macro_df.index.name = 'Date'
        
        # 将宏观特征与原始数据合并
        result = df.set_index('Date').join(macro_df)
        
        return result.reset_index()
    
    def generate_iceemdan_features(self, df):
        """生成ICEEMDAN分解特征"""
        close_values = df['Close'].values
        imfs = self.ceemdan.ceemdan(close_values)
        
        for i, imf in enumerate(imfs):
            df[f'IMF_{i}'] = imf
            
        df['trend'] = imfs[-1]
        return df
    
    def preprocess_features(self, df):
        """预处理特征（处理缺失值和异常值，但不做归一化）"""
        # 确保数据不为空
        if df.empty:
            self.logger.error("数据帧为空")
            return df
        
        # 检查缺失值情况
        missing_values = df.isnull().sum()
        missing_percent = (missing_values / len(df)) * 100
        
        # 记录缺失值情况
        self.logger.info("缺失值情况:")
        for col in df.columns:
            if missing_percent[col] > 0:
                self.logger.info(f"{col}: {missing_percent[col]:.2f}% ({missing_values[col]} 个缺失值)")
        
        # 处理无穷值
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # 检查并修正负值（仅对价格类数据）
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if col in df.columns and (df[col] < 0).any():
                self.logger.warning(f"检测到 {col} 列中有负值，修正为零")
                df[col] = df[col].clip(lower=0)
        
        # 分别处理不同类型的列
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # 1. 处理时间序列数据（如价格、成交量等）
        time_series_cols = [col for col in numeric_cols if col in price_cols or 'Volume' in col]
        for col in time_series_cols:
            if col in df.columns:
                # 使用前向填充，然后后向填充
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                
                # 如果仍有缺失值，使用移动平均填充
                if df[col].isnull().any():
                    self.logger.warning(f"{col} 仍有缺失值，使用移动平均填充")
                    ma = df[col].rolling(window=5, min_periods=1).mean()
                    df[col] = df[col].fillna(ma)
        
        # 2. 处理技术指标
        tech_indicator_cols = [col for col in numeric_cols if 
                              any(prefix in col for prefix in ['MA', 'momentum', 'volatility', 'BB', 'IMF', 'trend', 'returns'])]
        for col in tech_indicator_cols:
            if col in df.columns:
                # 检查缺失值数量
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    self.logger.info(f"{col} 有 {null_count} 个缺失值，开始填充")
                    
                    # 对于基于窗口计算的指标，前面的缺失值是正常的，使用前向填充不合适
                    # 首先检查是否是窗口开始的缺失值
                    first_valid_idx = df[col].first_valid_index()
                    if first_valid_idx is not None and first_valid_idx > 0:
                        # 对于窗口开始的缺失值，使用第一个有效值填充
                        first_valid_value = df.loc[first_valid_idx, col]
                        df.loc[:first_valid_idx, col] = first_valid_value
                        self.logger.info(f"{col} 的前 {first_valid_idx} 个值使用第一个有效值 {first_valid_value:.4f} 填充")
                    
                    # 对于中间的缺失值，使用线性插值
                    if df[col].isnull().any():
                        self.logger.info(f"{col} 仍有缺失值，使用线性插值填充")
                        df[col] = df[col].interpolate(method='linear')
                    
                    # 如果仍有缺失值（可能是末尾），使用后向填充
                    if df[col].isnull().any():
                        self.logger.info(f"{col} 仍有缺失值，使用后向填充")
                        df[col] = df[col].fillna(method='bfill')
                    
                    # 如果仍有缺失值，使用列均值填充
                    if df[col].isnull().any():
                        self.logger.warning(f"{col} 仍有缺失值，使用列均值填充")
                        df[col] = df[col].fillna(df[col].mean())
                    
                    # 验证填充结果
                    if df[col].isnull().any():
                        self.logger.error(f"{col} 填充后仍有 {df[col].isnull().sum()} 个缺失值")
                    else:
                        self.logger.info(f"{col} 填充完成")
        
        # 3. 处理宏观经济指标
        macro_cols = [col for col in numeric_cols if 
                     any(indicator in col for indicator in ['CPI', 'GDP', 'INFLATION', 'UNEMPLOYMENT'])]
        for col in macro_cols:
            if col in df.columns:
                # 宏观指标通常变化缓慢，使用前向填充
                df[col] = df[col].fillna(method='ffill')
                
                # 如果前面没有值可填充，使用后向填充
                df[col] = df[col].fillna(method='bfill')
                
                # 如果仍有缺失值，使用线性插值
                if df[col].isnull().any():
                    self.logger.warning(f"{col} 仍有缺失值，使用线性插值填充")
                    df[col] = df[col].interpolate(method='linear')
        
        # 4. 处理其他数值列
        other_numeric_cols = [col for col in numeric_cols if col not in time_series_cols + tech_indicator_cols + macro_cols]
        for col in other_numeric_cols:
            if col in df.columns:
                # 使用前向填充和后向填充
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                
                # 如果仍有缺失值，使用中位数填充（比均值更稳健）
                if df[col].isnull().any():
                    self.logger.warning(f"{col} 仍有缺失值，使用中位数填充")
                    df[col] = df[col].fillna(df[col].median())
        
        # 5. 处理非数值列（如日期、字符串等）
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        for col in non_numeric_cols:
            if col in df.columns and col != 'Date':  # 排除日期列
                # 使用最频繁值填充
                if df[col].isnull().any():
                    self.logger.warning(f"{col} 有缺失值，使用最频繁值填充")
                    most_frequent = df[col].mode()[0]
                    df[col] = df[col].fillna(most_frequent)
        
        # 检查是否仍有缺失值
        remaining_nulls = df.isnull().sum().sum()
        if remaining_nulls > 0:
            self.logger.warning(f"填充后仍有 {remaining_nulls} 个缺失值，删除这些行")
            df = df.dropna()
        
        if df.empty:
            self.logger.error("删除空值后数据帧为空")
            return df
        
        # 检测并处理异常值
        self.logger.info("检测异常值...")
        for col in numeric_cols:
            if col in df.columns:
                # 使用IQR方法检测异常值（比3倍标准差更稳健）
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                if len(outliers) > 0:
                    self.logger.warning(f"{col} 存在 {len(outliers)} 个异常值 ({(len(outliers)/len(df))*100:.2f}%)")
                    
                    # 对于价格数据，不处理异常值，因为可能是真实的价格波动
                    if col not in price_cols:
                        # 对于技术指标，使用截断法处理异常值
                        self.logger.info(f"对 {col} 的异常值进行截断处理")
                        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df
    
    def dimension_reduction(self, df):
        """特征降维"""
        self.logger.info("开始PCA降维...")
        
        # 保存日期列
        date_col = df['Date'] if 'Date' in df.columns else df.index
        
        # 选择数值型列
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # 应用PCA
        pca_result = self.pca.fit_transform(df[numeric_cols])
        
        # 创建新的DataFrame，包含日期列
        df_pca = pd.DataFrame(
            pca_result,
            columns=[f'PC{i+1}' for i in range(pca_result.shape[1])]
        )
        df_pca['Date'] = date_col  # 添加日期列
        
        # 记录降维信息
        self.logger.info(f"PCA降维后的特征数: {pca_result.shape[1]}")
        self.logger.info(f"PCA解释方差比:")
        for i, ratio in enumerate(self.pca.explained_variance_ratio_):
            cumsum = np.sum(self.pca.explained_variance_ratio_[:i+1])
            self.logger.info(f"PC{i+1}: {ratio:.4f} (累计: {cumsum:.4f})")
        
        # 将Date列移到第一列
        cols = df_pca.columns.tolist()
        cols = ['Date'] + [col for col in cols if col != 'Date']
        df_pca = df_pca[cols]
        
        return df_pca

    def normalize_features(self, df):
        """对特征进行归一化处理"""
        self.logger.info("对特征进行归一化处理...")
        
        # 保存非数值列
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        non_numeric_data = df[non_numeric_cols].copy() if non_numeric_cols else None
        
        # 获取数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 应用标准化
        normalized_data = self.scaler.fit_transform(df[numeric_cols])
        
        # 创建新的DataFrame
        df_normalized = pd.DataFrame(normalized_data, columns=numeric_cols, index=df.index)
        
        # 添加回非数值列
        if non_numeric_data is not None:
            df_normalized = pd.concat([non_numeric_data, df_normalized], axis=1)
        
        self.logger.info(f"归一化完成，处理了 {len(numeric_cols)} 个数值特征")
        
        return df_normalized

def main():
    """主函数"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # 所有需要处理的货币对
    pairs = ['CNYAUD', 'CNYEUR', 'CNYGBP', 'CNYJPY', 'CNYUSD']
    
    try:
        for pair in pairs:
            logger.info(f"\n开始处理货币对: {pair}")
            
            # 读取数据
            data_path = os.path.join('..', '..', 'try', 'data', f'{pair}.csv')
            logger.info(f"尝试读取文件: {os.path.abspath(data_path)}")
            
            if not os.path.exists(data_path):
                logger.error(f"文件不存在: {data_path}")
                continue
            
            # 读取数据
            df = pd.read_csv(data_path)
            
            # 创建特征工程实例
            engineer = FeatureEngineer()
            
            # 处理数据
            df_processed = engineer.process_data(df, pair)
            
            # 输出特征统计信息
            logger.info(f"\n{pair} 特征统计信息:")
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                mean = df_processed[col].mean()
                std = df_processed[col].std()
                if abs(mean) > 1e5 or std > 1e5:
                    logger.warning(f"{col} 可能存在尺度问题:")
                    logger.warning(f"均值: {mean}")
                    logger.warning(f"标准差: {std}")
            
            # 检查数据质量
            logger.info(f"\n{pair} 数据质量检查:")
            for col in numeric_cols:
                outliers = df_processed[abs(df_processed[col] - df_processed[col].mean()) > 3*df_processed[col].std()]
                if len(outliers) > 0:
                    logger.warning(f"{col} 存在 {len(outliers)} 个异常值")
                    logger.warning(f"范围: {outliers[col].min():.4f} - {outliers[col].max():.4f}")
            
            # 检查特征相关性
            corr_matrix = df_processed.corr()
            high_corr = np.where(np.abs(corr_matrix) > 0.95)
            high_corr = [(corr_matrix.index[x], corr_matrix.columns[y], corr_matrix.iloc[x, y])
                         for x, y in zip(*high_corr) if x != y]
            
            if high_corr:
                logger.warning(f"\n{pair} 高度相关的特征对:")
                for feat1, feat2, corr in high_corr:
                    logger.warning(f"{feat1} - {feat2}: {corr:.4f}")
            
            logger.info(f"{pair} 特征工程完成!")
            
    except Exception as e:
        logger.error(f"处理过程中出错: {str(e)}")
        raise

if __name__ == "__main__":
    main()
