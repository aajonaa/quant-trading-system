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
        
        self.n_components = 10  # PCA保留的组件数量
        
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
            
            # 4. 生成改进的交易信号
            self.logger.info("开始生成多周期加权交易信号")
            
            # 计算动量指标
            df['momentum_5'] = df['Close'].pct_change(5)  # 5天动量
            df['momentum_20'] = df['Close'].pct_change(20)  # 20天动量
            
            # 计算趋势指标
            df['ma_20'] = df['Close'].rolling(window=20).mean()
            df['trend'] = (df['Close'] - df['ma_20']) / df['ma_20']
            
            # 计算波动率和收益率指标
            df['volatility'] = df['returns'].rolling(window=20).std()
            df['returns_5'] = df['returns'].rolling(window=5).mean()  # 5日平均收益率
            
            # 多头信号条件
            long_conditions = (
                (df['momentum_5'] > 0.001) &  # 提高短期动量阈值
                (df['returns_5'] > 0.0003) &   # 提高近期收益要求
                (
                    (df['momentum_20'] > 0.0005) |   # 提高中期动量要求
                    (df['trend'] > 0.002)            # 提高趋势要求
                )
            )
            
            # 空头信号条件
            short_conditions = (
                (df['momentum_5'] < -0.001) &  # 提高短期动量阈值
                (df['returns_5'] < -0.0003) &   # 提高近期收益要求
                (
                    (df['momentum_20'] < -0.0005) |  # 提高中期动量要求
                    (df['trend'] < -0.002)           # 提高趋势要求
                )
            )
            
            # 使用 loc 进行赋值以避免警告
            df.loc[long_conditions, 'signal'] = 1
            df.loc[short_conditions, 'signal'] = -1
            df.loc[~(long_conditions | short_conditions), 'signal'] = 0
            
            # 添加波动率过滤（适度过滤）
            high_volatility = df['volatility'] > df['volatility'].rolling(window=50).mean() * 2
            df.loc[high_volatility, 'signal'] = 0
            
            # 添加止损条件
            STOP_LOSS_THRESHOLD = 0.02  # 恢复到2%止损线
            for i in range(1, len(df)):
                if df['signal'].iloc[i-1] != 0:  # 如果前一天有持仓
                    returns = df['returns'].iloc[i]
                    if abs(returns) > STOP_LOSS_THRESHOLD:  # 如果超过止损线
                        df.loc[df.index[i], 'signal'] = 0
            
            # 验证信号质量
            signal_stats = self._validate_signals(df)
            if signal_stats is None:
                return None
            
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
                if col != 'signal':  # 排除信号列
                    df[col] = df[col].ffill().bfill()  # 使用新的填充方法
            
            # 7. 输出信号统计
            signal_dist = df['signal'].value_counts(normalize=True)
            self.logger.info("\n交易信号分布:")
            for signal in sorted(signal_dist.index):
                self.logger.info(f"信号 {signal}: {signal_dist[signal]*100:.2f}%")
            
            self.logger.info("\n信号统计:")
            self.logger.info(f"信号1的平均收益率: {df[df['signal'] == 1]['returns'].mean():.4f}")
            self.logger.info(f"信号1的平均波动率: {df[df['signal'] == 1]['volatility'].mean():.4f}")
            self.logger.info(f"信号-1的平均收益率: {df[df['signal'] == -1]['returns'].mean():.4f}")
            self.logger.info(f"信号-1的平均波动率: {df[df['signal'] == -1]['volatility'].mean():.4f}")
            
            # 添加ICEEMDAN分解特征
            df = self.add_iceemdan_features(df)
            if df is None:
                return None
                
            # 处理所有特征的缺失值
            numeric_cols = df.select_dtypes(include=[np.float64, np.int64]).columns
            for col in numeric_cols:
                if col != 'signal':  # 排除信号列
                    df[col] = df[col].ffill().bfill()
            
            # 处理无穷值和极端值
            def clean_and_scale_features(df):
                """清理特征，但不进行标准化"""
                # 创建副本避免警告
                df = df.copy()
                
                # 价格和信号特征列表
                price_cols = ['Open', 'High', 'Low', 'Close']
                essential_cols = price_cols + ['signal']
                
                # 处理非基础特征
                for col in df.columns:
                    if col not in essential_cols:
                        # 替换无穷值为0
                        df[col] = df[col].replace([np.inf, -np.inf], 0)
                        
                        # 使用前向填充处理缺失值
                        df[col] = df[col].fillna(method='ffill')
                        # 使用后向填充处理剩余的缺失值
                        df[col] = df[col].fillna(method='bfill')
                        # 如果仍有缺失值，用0填充
                        df[col] = df[col].fillna(0)
                        
                        # 限制极端值范围(使用分位数)
                        if df[col].std() != 0:  # 只处理非常数列
                            q1 = df[col].quantile(0.01)
                            q3 = df[col].quantile(0.99)
                            df[col] = df[col].clip(lower=q1, upper=q3)
                
                return df
                
            # 修复pct_change警告
            def calculate_changes(df):
                for col in df.columns:
                    if '_CHANGE' in col:
                        # 先处理缺失值,再计算变化率
                        series = df[col].fillna(method='ffill')
                        df[col] = series.pct_change(fill_method=None)
                return df
                
            # 优化特征选择
            def select_features(df):
                # 定义必须保留的特征
                essential_features = ['Close', 'signal']
                
                # 计算相关性矩阵
                corr_matrix = df.corr().abs()
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                
                # 找出高度相关的特征对
                high_corr_pairs = []
                for col in upper.columns:
                    highly_correlated = upper[col][upper[col] > 0.95].index.tolist()
                    for corr_col in highly_correlated:
                        if col not in essential_features and corr_col not in essential_features:
                            high_corr_pairs.append((col, corr_col))
                
                # 从每对高度相关的特征中删除一个
                to_drop = set()
                for feat1, feat2 in high_corr_pairs:
                    if feat1 not in essential_features and feat2 not in essential_features:
                        # 保留名字较短的特征（通常是更基础的特征）
                        if len(feat1) > len(feat2):
                            to_drop.add(feat1)
                        else:
                            to_drop.add(feat2)
                
                # 删除选定的特征
                return df.drop(columns=list(to_drop))
                
            # 应用处理函数
            df = clean_and_scale_features(df)
            df = calculate_changes(df)
            df = select_features(df)
            
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

    def _validate_signals(self, df):
        """验证信号质量"""
        try:
            # 计算每种信号的统计信息
            signal_stats = {}
            for signal in df['signal'].unique():
                mask = df['signal'] == signal
                signal_data = df[mask]
                
                stats = {
                    'count': len(signal_data),
                    'returns_mean': signal_data['returns'].mean(),
                    'volatility_mean': signal_data['volatility'].mean()
                }
                signal_stats[signal] = stats
                
                # 输出统计信息
                self.logger.info(f"\n信号 {signal} 统计:")
                self.logger.info(f"数量: {stats['count']}")
                self.logger.info(f"平均收益率: {stats['returns_mean']:.4f}")
                self.logger.info(f"平均波动率: {stats['volatility_mean']:.4f}")
            
            # 计算信号转换频率
            signal_changes = (df['signal'] != df['signal'].shift(1)).sum()
            conversion_rate = signal_changes / len(df)
            self.logger.info(f"\n信号转换频率: {conversion_rate:.4f}")
            
            return signal_stats
            
        except Exception as e:
            self.logger.error(f"验证信号时发生错误: {str(e)}")
            return None

    def apply_pca(self, df):
        """应用PCA降维，在PCA前进行标准化"""
        try:
            # 选择数值型特征列，排除基础特征
            essential_cols = ['signal']
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            feature_cols = [col for col in numeric_cols if col not in essential_cols]
            
            if not feature_cols:
                self.logger.error("没有可用于PCA的特征")
                return None
            
            features = df[feature_cols]
            
            # 检查数据是否包含无穷值或NaN
            if np.any(np.isinf(features)) or np.any(np.isnan(features)):
                self.logger.warning("数据中包含无穷值或NaN，进行清理...")
                features = features.replace([np.inf, -np.inf], 0)
                features = features.fillna(0)
            
            # 检查是否有常数列
            constant_cols = features.columns[features.std() == 0]
            if len(constant_cols) > 0:
                self.logger.warning(f"删除常数列: {constant_cols}")
                features = features.drop(columns=constant_cols)
            
            # 检查剩余特征数量
            if len(features.columns) < self.n_components:
                self.n_components = len(features.columns)
                self.logger.warning(f"调整PCA组件数量为: {self.n_components}")
            
            # 对特征进行标准化处理
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            # 使用标准化后的数据进行PCA
            pca = PCA(n_components=self.n_components)
            pca_features = pca.fit_transform(scaled_features)
            
            # 创建PCA特征的DataFrame
            pca_df = pd.DataFrame(
                pca_features,
                columns=[f'PCA_{i+1}' for i in range(self.n_components)],
                index=df.index
            )
            
            # 添加基础特征（使用原始数据）
            for col in essential_cols:
                pca_df[col] = df[col]
            
            # 计算并记录解释方差比
            explained_variance_ratio = pca.explained_variance_ratio_
            cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
            
            self.logger.info("\nPCA解释方差比:")
            for i, (var_ratio, cum_ratio) in enumerate(zip(explained_variance_ratio, cumulative_variance_ratio)):
                self.logger.info(f"PC{i+1}: {var_ratio:.4f} (累计: {cum_ratio:.4f})")
            
            # 记录总解释方差
            total_variance = sum(explained_variance_ratio)
            self.logger.info(f"\n总解释方差比: {total_variance:.4f}")
            
            # 记录每个主成分的特征贡献
            feature_importance = pd.DataFrame(
                pca.components_.T,
                columns=[f'PC{i+1}' for i in range(self.n_components)],
                index=features.columns
            )
            
            self.logger.info("\n特征对主成分的贡献:")
            for pc in feature_importance.columns[:3]:  # 只显示前3个主成分
                top_features = feature_importance[pc].abs().nlargest(5)
                self.logger.info(f"\n{pc}的主要特征:")
                for feat, value in top_features.items():
                    self.logger.info(f"{feat}: {value:.4f}")
            
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

    def _add_macro_features(self, df, pair):
        """添加宏观经济特征，包括指标差异、比率和价差"""
        try:
            # 货币到国家的映射
            currency_map = {
                'CNY': 'CN',
                'EUR': 'EU',
                'GBP': 'UK',
                'USD': 'US',
                'JPY': 'JP'
            }
            
            # 获取基础货币和报价货币的国家代码
            base_country = currency_map.get(pair[:3])
            quote_country = currency_map.get(pair[3:])
            
            if not base_country or not quote_country:
                self.logger.error(f"无法映射货币对 {pair} 到对应国家")
                return df
            
            # 定义宏观指标映射
            indicator_map = {
                'CPI': {
                    'file': 'CPI',
                    'desc': '消费者价格指数',
                    'unit': '指数值'
                },
                'INFLATION': {
                    'file': 'INFLATION',
                    'desc': '通货膨胀率',
                    'unit': '百分比'
                },
                'REAL_GDP': {
                    'file': 'REAL_GDP',
                    'desc': '实际GDP',
                    'unit': '真实值'
                },
                'UNEMPLOYMENT': {
                    'file': 'UNEMPLOYMENT',
                    'desc': '失业率',
                    'unit': '百分比'
                }
            }
            
            # 存储所有宏观数据
            macro_data = {}
            
            # 加载两个国家的所有指标数据
            for country in [base_country, quote_country]:
                for ind_name, ind_info in indicator_map.items():
                    file_path = self.macro_dir / f"{country}_{ind_info['file']}.csv"
                    if file_path.exists():
                        try:
                            # 读取数据
                            data = pd.read_csv(file_path)
                            data['Date'] = pd.to_datetime(data['date'])
                            data.set_index('Date', inplace=True)
                            
                            # 使用线性插值填充日度数据
                            filled_data = self._interpolate_macro_data(data)
                            if filled_data is not None:
                                macro_data[f"{country}_{ind_name}"] = filled_data
                                
                        except Exception as e:
                            self.logger.error(f"处理{country}的{ind_name}数据失败: {str(e)}")
                            continue
            
            # 计算指标差异特征
            for ind_name in indicator_map.keys():
                base_key = f"{base_country}_{ind_name}"
                quote_key = f"{quote_country}_{ind_name}"
                
                if base_key in macro_data and quote_key in macro_data:
                    try:
                        # 对齐数据到交易日期
                        base_aligned = macro_data[base_key].reindex(df.index).ffill().bfill()
                        quote_aligned = macro_data[quote_key].reindex(df.index).ffill().bfill()
                        
                        # 计算三种差异指标
                        df[f'MACRO_{ind_name}_DIFF'] = base_aligned - quote_aligned
                        df[f'MACRO_{ind_name}_RATIO'] = base_aligned / quote_aligned
                        df[f'MACRO_{ind_name}_SPREAD'] = (base_aligned - quote_aligned) / quote_aligned
                        
                        # 添加滚动变化率
                        for window in [5, 20]:
                            diff_col = f'MACRO_{ind_name}_DIFF'
                            df[f'{diff_col}_CHANGE_{window}D'] = df[diff_col].pct_change(window)
                            
                    except Exception as e:
                        self.logger.error(f"计算{ind_name}差异指标失败: {str(e)}")
                        continue
            
            # 处理异常值和缺失值
            macro_cols = [col for col in df.columns if col.startswith('MACRO_')]
            for col in macro_cols:
                # 处理无穷值
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                
                # 使用前向填充处理缺失值
                df[col] = df[col].fillna(method='ffill')
                df[col] = df[col].fillna(method='bfill')
                df[col] = df[col].fillna(0)
                
                # 处理极端值
                if df[col].std() != 0:
                    q1 = df[col].quantile(0.01)
                    q3 = df[col].quantile(0.99)
                    df[col] = df[col].clip(lower=q1, upper=q3)
            
            # 记录添加的特征信息
            self.logger.info(f"\n为{pair}添加的宏观经济特征:")
            for ind_name in indicator_map.keys():
                feature_cols = [col for col in df.columns if f'MACRO_{ind_name}' in col]
                if feature_cols:
                    self.logger.info(f"\n{indicator_map[ind_name]['desc']}相关特征:")
                    for col in feature_cols:
                        self.logger.info(f"- {col}")
                    
            return df
            
        except Exception as e:
            self.logger.error(f"添加宏观经济特征失败: {str(e)}")
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