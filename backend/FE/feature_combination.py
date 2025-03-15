import pandas as pd
import numpy as np
from pathlib import Path
import logging
from PyEMD import CEEMDAN
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

class FeatureEngineer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # 获取当前文件所在目录
        current_dir = Path(__file__).parent
        self.logger.info(f"当前目录: {current_dir.absolute()}")
        
        # 获取项目根目录（backend的父目录）
        root_dir = current_dir.parent.parent
        self.logger.info(f"根目录: {root_dir.absolute()}")
        
        # 设置数据源路径为项目根目录下的try/data
        self.data_dir = root_dir / "try" / "data"
        self.logger.info(f"数据目录: {self.data_dir.absolute()}")
        
        # 检查data目录是否存在
        if not self.data_dir.exists():
            self.logger.error(f"data目录不存在: {self.data_dir}")
            # 列出try目录下的所有文件和文件夹
            try_dir = root_dir / "try"
            if try_dir.exists():
                self.logger.info(f"try目录内容: {list(try_dir.iterdir())}")
        
        # 输出目录设置为FE文件夹
        self.output_dir = current_dir
        
        self.ceemdan = CEEMDAN()
        
        # 修改宏观数据目录路径
        self.macro_dir = root_dir / "try" / "macro_data"  # 改为try下的macro_data
        if not self.macro_dir.exists():
            self.macro_dir.mkdir(parents=True)
            self.logger.info(f"创建宏观数据目录: {self.macro_dir}")
        
        self.n_components = 20  # PCA保留的组件数量
        
    def load_data(self, pair):
        """加载原始数据"""
        try:
            file_path = self.data_dir / f"{pair}.csv"
            self.logger.info(f"尝试加载文件: {file_path.absolute()}")
            
            # 检查文件是否存在
            if not file_path.exists():
                # 列出try目录下的所有文件
                if self.data_dir.exists():
                    self.logger.info(f"try目录内容: {list(self.data_dir.iterdir())}")
                self.logger.error(f"找不到文件: {file_path}")
                return None
                
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            self.logger.info(f"成功加载数据，形状: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"加载数据失败: {str(e)}")
            return None
            
    def create_features(self, df):
        """创建特征"""
        try:
            # 获取货币对名称，但保持时间索引名称为'Date'
            pair = df.name if hasattr(df, 'name') else None
            if pair is None:
                # 从DataFrame的属性中获取货币对名称
                pair = getattr(df, 'currency_pair', None)
                
                if pair is None or pair == "":
                    self.logger.error("无法确定货币对名称")
                    return None
                
                # 设置DataFrame的name属性，但不改变索引名称
                df.name = pair
            
            self.logger.info(f"处理货币对: {pair}")
            
            # 1. 创建基础特征
            df['returns'] = df['Close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=20).std()
            
            # 2. 处理基础特征的缺失值
            df['returns'] = df['returns'].fillna(0)
            df['volatility'] = df['volatility'].ffill().bfill()
            
            # 3. 添加宏观经济特征
            self.logger.info(f"开始添加宏观经济特征: {pair}")
            df = self._add_macro_features(df, pair)
            self.logger.info(f"完成添加宏观经济特征")
            
            # 4. 生成交易信号
            df['reSignal'] = 0  # 默认为中性信号
            
            # 使用固定阈值生成信号
            RETURN_THRESHOLD = 0.001  # 0.1%
            
            # 生成信号
            df.loc[df['returns'].shift(-1) > RETURN_THRESHOLD, 'reSignal'] = 1
            df.loc[df['returns'].shift(-1) < -RETURN_THRESHOLD, 'reSignal'] = -1

            # 最后一天的信号设为0（因为没有下一天的return）
            df.iloc[-1, df.columns.get_loc('reSignal')] = 0
            
            # 4. 添加动量特征
            self.add_momentum_features(df)
            
            # 5. 添加技术指标
            for window in [5, 10, 20, 50]:
                df[f'MA{window}'] = df['Close'].rolling(window=window).mean()
                df[f'trend_{window}'] = (df['Close'] - df[f'MA{window}']) / df[f'MA{window}']
            
            middle, upper, lower = self.calculate_bollinger_bands(df['Close'])
            df['bb_width'] = (upper - lower) / middle
            df['bb_pos'] = (df['Close'] - lower) / (upper - lower)
            
            # 6. 处理所有特征的缺失值
            numeric_cols = df.select_dtypes(include=[np.float64, np.int64]).columns
            for col in numeric_cols:
                if col != 'reSignal':  # 排除信号列
                    df[col] = df[col].ffill().bfill()  # 使用新的填充方法
            
            # 7. 输出信号统计
            signal_dist = df['reSignal'].value_counts(normalize=True)
            self.logger.info("\n交易信号分布:")
            for signal in sorted(signal_dist.index):
                self.logger.info(f"信号 {signal}: {signal_dist[signal]*100:.2f}%")
            
            self.logger.info("\n信号统计:")
            self.logger.info(f"信号1的平均收益率: {df[df['reSignal'] == 1]['returns'].mean():.4f}")
            self.logger.info(f"信号1的平均波动率: {df[df['reSignal'] == 1]['volatility'].mean():.4f}")
            self.logger.info(f"信号-1的平均收益率: {df[df['reSignal'] == -1]['returns'].mean():.4f}")
            self.logger.info(f"信号-1的平均波动率: {df[df['reSignal'] == -1]['volatility'].mean():.4f}")
            
            # 添加ICEEMDAN分解特征
            df = self.add_iceemdan_features(df)
            if df is None:
                return None
                
            # 处理所有特征的缺失值
            numeric_cols = df.select_dtypes(include=[np.float64, np.int64]).columns
            for col in numeric_cols:
                if col != 'reSignal':  # 排除信号列
                    df[col] = df[col].ffill().bfill()
            
            return df
            
        except Exception as e:
            self.logger.error(f"创建特征失败: {str(e)}")
            return None

    def calculate_rsi(self, prices, period=14):
        """计算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """计算MACD"""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram

    def calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """计算布林带"""
        middle = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        return middle, upper, lower

    def calculate_atr(self, high, low, close, period=14):
        """计算ATR"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def calculate_cci(self, high, low, close, period=20):
        """计算CCI"""
        tp = (high + low + close) / 3
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        return (tp - sma) / (0.015 * mad)

    def add_momentum_features(self, df):
        """添加动量特征"""
        try:
            # 价格动量
            for period in [5, 10, 21, 63]:
                # 价格变化率
                df[f'momentum_{period}'] = df['Close'].pct_change(period)

                # 价格加速度
                df[f'acceleration_{period}'] = df[f'momentum_{period}'] - df[f'momentum_{period}'].shift(1)
                
                # 波动率动量
                vol = df['returns'].rolling(window=period).std()
                df[f'volatility_momentum_{period}'] = vol.pct_change(period)
            
            # TSI (True Strength Index)
            def calculate_tsi(close, r=25, s=13):
                diff = close.diff()
                abs_diff = abs(diff)
                
                smooth1 = diff.ewm(span=r, adjust=False).mean()
                smooth2 = smooth1.ewm(span=s, adjust=False).mean()
                abs_smooth1 = abs_diff.ewm(span=r, adjust=False).mean()
                abs_smooth2 = abs_smooth1.ewm(span=s, adjust=False).mean()
                
                return (smooth2 / (abs_smooth2 + 1e-10)) * 100

            df['TSI'] = calculate_tsi(df['Close'])
            
            # 动量分位数
            for period in [5, 10, 21, 63]:
                rolling_returns = df['returns'].rolling(window=period)
                df[f'momentum_quantile_{period}'] = rolling_returns.apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1] 
                    if len(x) > 0 else np.nan
                )
            
            return df
            
        except Exception as e:
            self.logger.error(f"添加动量特征失败: {str(e)}")
            return None

    def add_seasonality_features(self, df):
        """添加季节性特征"""
        try:
            from statsmodels.tsa.seasonal import STL
            
            # 对收盘价进行STL分解
            stl = STL(df['Close'], period=21)
            result = stl.fit()
            
            # 趋势成分
            df['trend'] = result.trend
            df['trend_strength'] = abs(df['Close'] - result.trend) / df['Close']
            
            # 季节性成分
            df['seasonal'] = result.seasonal
            df['seasonal_strength'] = abs(result.seasonal) / df['Close']
            
            # 残差成分
            df['residual'] = result.resid
            df['residual_strength'] = abs(result.resid) / df['Close']
            
            return df
            
        except Exception as e:
            self.logger.error(f"添加季节性特征失败: {str(e)}")
            return None

    def add_iceemdan_features(self, df):
        """添加ICEEMDAN分解特征"""
        try:
            # 准备价格数据
            close_prices = df['Close'].values
            
            # 执行ICEEMDAN分解
            imfs = self.ceemdan(close_prices)
            
            # 创建一个DataFrame来存储IMF特征
            imf_features = pd.DataFrame(index=df.index)
            
            # 添加每个IMF分量作为特征
            for i, imf in enumerate(imfs):
                # 添加原始IMF
                imf_features[f'imf_{i}'] = imf
                
                # 计算滚动统计量
                imf_series = pd.Series(imf, index=df.index)
                
                # 计算波动率 (20天滚动标准差)
                imf_features[f'imf_{i}_volatility'] = imf_series.rolling(window=20, min_periods=1).std()
                
                # 计算能量 (20天滚动平方和的平均)
                imf_features[f'imf_{i}_energy'] = (imf_series ** 2).rolling(window=20, min_periods=1).mean()
                
                # 计算与原始价格的相关性 (20天滚动相关)
                imf_features[f'imf_{i}_correlation'] = imf_series.rolling(window=20, min_periods=1).corr(df['Close'])
            
            # 计算残差
            residual = close_prices - np.sum(imfs, axis=0)
            imf_features['iceemdan_residual'] = residual
            
            # 计算IMF的组合特征
            imf_features['imf_total_energy'] = sum(imf_features[f'imf_{i}_energy'] for i in range(len(imfs)))
            imf_features['imf_weighted_sum'] = sum(imf_features[f'imf_{i}'] * (1 / (i + 1)) for i in range(len(imfs)))
            
            # 前向填充缺失值
            imf_features = imf_features.fillna(method='ffill')
            # 后向填充剩余的缺失值
            imf_features = imf_features.fillna(method='bfill')
            
            # 将IMF特征合并到原始DataFrame
            df = pd.concat([df, imf_features], axis=1)
            
            self.logger.info(f"添加了 {len(imfs)} 个IMF分量特征")
            return df
            
        except Exception as e:
            self.logger.error(f"添加ICEEMDAN分解特征失败: {str(e)}")
            return None

    def _interpolate_macro_data(self, data):
        """使用线性回归方法将低频数据转换为日度数据"""
        try:
            # 创建完整的日期范围
            date_range = pd.date_range(start=data.index.min(), end=data.index.max(), freq='D')
            
            # 获取数值列名
            value_col = data.select_dtypes(include=['float64', 'int64']).columns[0]
            
            # 创建包含所有日期的DataFrame
            daily_data = pd.DataFrame(index=date_range)
            daily_data['time_idx'] = (daily_data.index - daily_data.index[0]).days
            daily_data[value_col] = np.nan
            
            # 将原始数据合并到日度数据中
            daily_data.loc[data.index, value_col] = data[value_col]
            
            # 对每个非空区间进行线性回归
            filled_data = pd.Series(index=daily_data.index)
            non_null_dates = daily_data[daily_data[value_col].notna()].index
            
            if len(non_null_dates) >= 2:
                # 将日期转换为时间戳以进行回归
                X = daily_data.loc[non_null_dates, 'time_idx'].values.reshape(-1, 1)
                y = daily_data.loc[non_null_dates, value_col].values
                
                # 训练回归模型
                reg = LinearRegression()
                reg.fit(X, y)
                
                # 对所有日期进行预测
                X_all = daily_data['time_idx'].values.reshape(-1, 1)
                filled_values = reg.predict(X_all)
                
                # 创建填充后的序列
                filled_data = pd.Series(filled_values, index=daily_data.index)
                
                # 保持原始数据点不变
                filled_data[non_null_dates] = daily_data.loc[non_null_dates, value_col]
                
                return filled_data
            
            return None
            
        except Exception as e:
            self.logger.error(f"宏观数据插值失败: {str(e)}")
            return None

    def _add_macro_features(self, df, pair):
        """添加宏观经济特征"""
        try:
            # 获取基础货币和报价货币的国家代码
            base_currency = pair[:3]  # CNY
            quote_currency = pair[3:] # EUR, USD等
            
            # 国家代码映射
            country_map = {
                'CNY': 'CN',
                'USD': 'US',
                'EUR': 'EU',
                'GBP': 'UK',  # 注意英国使用UK
                'JPY': 'JP',
            }
            
            base_country = country_map.get(base_currency)
            quote_country = country_map.get(quote_currency)
            
            if not base_country or not quote_country:
                self.logger.warning(f"货币对 {pair} 包含未知的货币代码")
                return df
            
            # 宏观经济指标
            indicators = {
                'CPI': '消费者价格指数',
                'REAL_GDP': '实际国内生产总值',
                'INFLATION':'通货膨胀率',
                'REAL_SALES':'实际销售额',
                'UNEMPLOYMENT':'失业率'
                # 可以添加更多指标
            }
            
            start_date = df.index.min()
            
            # 处理每个国家的每个指标
            for country in [base_country, quote_country]:
                for indicator in indicators.keys():
                    try:
                        # 读取数据文件
                        file_path = self.data_dir.parent / "macro_data" / f"{country}_{indicator}.csv"
                        if not file_path.exists():
                            self.logger.warning(f"找不到{indicators[indicator]}数据文件: {file_path}")
                            continue
                        
                        # 读取并处理数据
                        data = pd.read_csv(file_path)
                        data['date'] = pd.to_datetime(data['date'])
                        data.set_index('date', inplace=True)
                        
                        # 使用回归方法生成日度数据
                        daily_data = self._interpolate_macro_data(data)
                        if daily_data is None:
                            continue
                        
                        # 对齐数据到交易日
                        aligned_data = daily_data.reindex(df.index)
                        
                        # 添加基础特征
                        col_name = f'{country}_{indicator}'
                        df[col_name] = aligned_data
                        
                        # 添加变化率（使用日度数据计算）
                        df[f'{col_name}_CHANGE'] = df[col_name].pct_change()

                        # 添加波动率（基于日度数据）
                        df[f'{col_name}_VOL'] = df[col_name].rolling(30).std()
                        
                    except Exception as e:
                        self.logger.error(f"处理{country}_{indicator}数据失败: {str(e)}")
                        continue
            
            # 计算国家间差值特征
            for indicator in indicators.keys():
                base_col = f'{base_country}_{indicator}'
                quote_col = f'{quote_country}_{indicator}'
                if base_col in df.columns and quote_col in df.columns:
                    df[f'{indicator}_DIFF'] = df[base_col] - df[quote_col]
                    df[f'{indicator}_RATIO'] = df[base_col] / df[quote_col]
                    df[f'{indicator}_SPREAD'] = (df[base_col] - df[quote_col]) / df[quote_col]
            
            # 处理可能的极端值
            numeric_cols = df.select_dtypes(include=[np.float64, np.int64]).columns
            for col in numeric_cols:
                if col != 'reSignal':
                    df[col] = df[col].clip(lower=df[col].quantile(0.01), 
                                         upper=df[col].quantile(0.99))
            
            # 输出添加的特征信息
            macro_cols = [col for col in df.columns if any(c in col for c in [base_country, quote_country])]
            self.logger.info(f"成功添加 {len(macro_cols)} 个宏观经济特征")
            self.logger.info(f"新增特征: {macro_cols}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"添加宏观经济特征失败: {str(e)}")
            return df

    def apply_pca(self, df):
        """应用PCA降维"""
        try:
            # 选择数值型特征列，排除reSignal列
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            numeric_cols = numeric_cols[numeric_cols != 'reSignal']
            features = df[numeric_cols]
            
            # 标准化
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            # 应用PCA
            pca = PCA(n_components=self.n_components)
            pca_features = pca.fit_transform(scaled_features)
            
            # 创建PCA特征的DataFrame
            pca_df = pd.DataFrame(
                pca_features,
                columns=[f'PCA_{i+1}' for i in range(self.n_components)],
                index=df.index
            )
            
            # 添加reSignal列
            pca_df['reSignal'] = df['reSignal']
            
            # 计算解释方差比
            explained_variance_ratio = pca.explained_variance_ratio_
            cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
            
            self.logger.info("PCA解释方差比:")
            for i, (var_ratio, cum_ratio) in enumerate(zip(explained_variance_ratio, cumulative_variance_ratio)):
                self.logger.info(f"PC{i+1}: {var_ratio:.4f} (累计: {cum_ratio:.4f})")
            
            return pca_df
            
        except Exception as e:
            self.logger.error(f"PCA降维失败: {str(e)}")
            return None

    def process_all_pairs(self):
        """处理所有货币对"""
        try:
            pairs = ["CNYEUR", "CNYGBP", "CNYJPY", "CNYUSD"]
            
            for pair in pairs:
                self.logger.info(f"处理 {pair}...")
                
                try:
                    # 加载数据
                    df = self.load_data(pair)
                    if df is None:
                        continue
                    
                    # 设置货币对属性，但保持索引名称为'Date'
                    df.currency_pair = pair
                    
                    # 记录原始数据信息
                    self.logger.info(f"原始数据统计:")
                    self.logger.info(f"价格范围: {df['Close'].min():.4f} - {df['Close'].max():.4f}")
                    
                    # 创建特征
                    df_processed = self.create_features(df)
                    if df_processed is None:
                        continue
                    
                    # 数据质量检查
                    self.check_data_quality(df_processed, pair)
                    
                    # 保存原始处理后的数据
                    output_path = self.output_dir / f"{pair}_processed.csv"
                    df_processed.to_csv(output_path, mode='w')  # 使用mode='w'覆盖现有文件
                    self.logger.info(f"已保存处理后的数据到: {output_path}")
                    
                    # 应用PCA并保存
                    self.logger.info("开始PCA降维...")
                    df_pca = self.apply_pca(df_processed)
                    if df_pca is not None:
                        pca_path = self.output_dir / f"{pair}_PCA.csv"
                        df_pca.to_csv(pca_path, mode='w')  # 使用mode='w'覆盖现有文件
                        self.logger.info(f"已保存PCA处理后的数据到: {pca_path}")
                        self.logger.info(f"PCA特征数量: {len(df_pca.columns)}")
                    
                    # 记录特征信息
                    self.logger.info(f"原始特征数量: {len(df_processed.columns)}")
                    self.logger.info(f"数据长度: {len(df_processed)}")
                    
                except PermissionError:
                    self.logger.error(f"无法写入文件，请确保文件未被其他程序占用: {pair}")
                    continue
                except Exception as e:
                    self.logger.error(f"处理{pair}时发生错误: {str(e)}")
                    continue
            
            return True
            
        except Exception as e:
            self.logger.error(f"处理数据失败: {str(e)}")
            return False

    def check_data_quality(self, df, pair):
        """检查数据质量"""
        self.logger.info(f"\n{pair} 数据质量检查:")
        
        # 1. 检查缺失值
        na_count = df.isna().sum()
        if na_count.any():
            self.logger.warning(f"存在缺失值的列:\n{na_count[na_count > 0]}")
        
        # 2. 检查无穷值
        inf_count = df.isin([np.inf, -np.inf]).sum()
        if inf_count.any():
            self.logger.warning(f"存在无穷值的列:\n{inf_count[inf_count > 0]}")
        
        # 3. 检查异常值 (使用3个标准差作为阈值)
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                mean = df[col].mean()
                std = df[col].std()
                outliers = df[col][(df[col] > mean + 3*std) | (df[col] < mean - 3*std)]
                if len(outliers) > 0:
                    self.logger.warning(f"{col} 存在 {len(outliers)} 个异常值")
                    self.logger.warning(f"范围: {outliers.min():.4f} - {outliers.max():.4f}")
        
        # 4. 检查特征相关性
        high_corr_pairs = []
        corr_matrix = df.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > 0.95:  # 高度相关阈值
                    high_corr_pairs.append(
                        (corr_matrix.index[i], corr_matrix.columns[j], 
                         corr_matrix.iloc[i, j])
                    )
        
        if high_corr_pairs:
            self.logger.warning("高度相关的特征对:")
            for feat1, feat2, corr in high_corr_pairs:
                self.logger.warning(f"{feat1} - {feat2}: {corr:.4f}")
        
        # 5. 检查特征分布
        self.logger.info("\n特征统计信息:")
        stats = df.describe()
        for col in stats.columns:
            if abs(stats[col]['mean']) > 1e3 or abs(stats[col]['std']) > 1e3:
                self.logger.warning(f"{col} 可能存在尺度问题:")
                self.logger.warning(f"均值: {stats[col]['mean']:.4f}")
                self.logger.warning(f"标准差: {stats[col]['std']:.4f}")
        
        # 6. 检查时间连续性
        time_diff = df.index.to_series().diff()
        if time_diff.max() > pd.Timedelta(days=2):
            self.logger.warning("存在时间间隔大于2天的数据")
            gaps = time_diff[time_diff > pd.Timedelta(days=2)]
            self.logger.warning(f"最大间隔: {gaps.max()}")

    def handle_outliers(self, df, columns, n_std=3):
        """处理异常值"""
        for col in columns:
            if df[col].dtype in [np.float64, np.int64]:
                mean = df[col].mean()
                std = df[col].std()
                df[col] = df[col].clip(mean - n_std*std, mean + n_std*std)
        return df

def main():
    """主函数"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建特征工程实例
    engineer = FeatureEngineer()
    
    # 处理所有数据
    if engineer.process_all_pairs():
        logging.info("特征工程完成!")
    else:
        logging.error("特征工程失败!")

if __name__ == "__main__":
    main() 