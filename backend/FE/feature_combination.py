import pandas as pd
import numpy as np
from pathlib import Path
import logging
from PyEMD import CEEMDAN
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
        """创建特征并生成交易信号"""
        try:
            # 设置DataFrame的货币对属性
            pair = df.name if hasattr(df, 'name') else None
            if pair is None:
                pair = getattr(df, 'currency_pair', None)
                if pair is None:
                    self.logger.error("无法确定货币对名称")
                    return None
                df.name = pair
            
            self.logger.info(f"处理货币对: {pair}")
            
            # 1. ICEEMDAN分解
            close_values = df['Close'].values
            imfs = self.ceemdan.ceemdan(close_values)
            
            # 添加IMF分量作为特征
            for i, imf in enumerate(imfs):
                df[f'IMF_{i}'] = imf
            
            # 计算趋势分量(最后一个IMF)
            df['trend'] = imfs[-1]
            
            # 2. 基础价格特征
            df['returns'] = df['Close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=20).std()
            
            # 3. 添加宏观经济特征
            df = self._add_macro_features(df, pair)
            
            # 4. 技术指标
            # 多周期移动平均
            for window in [5, 10, 20, 50]:
                df[f'MA{window}'] = df['Close'].rolling(window=window).mean()
                df[f'MA_dist_{window}'] = (df['Close'] - df[f'MA{window}']) / df[f'MA{window}']
            
            # RSI指标
            df['RSI'] = self.calculate_rsi(df['Close'], 14)
            
            # MACD指标
            macd, signal, hist = self.calculate_macd(df['Close'])
            df['MACD'] = macd
            df['MACD_signal'] = signal
            df['MACD_hist'] = hist
            
            # 布林带
            df['BB_middle'] = df['Close'].rolling(window=20).mean()
            df['BB_upper'] = df['BB_middle'] + 2 * df['Close'].rolling(window=20).std()
            df['BB_lower'] = df['BB_middle'] - 2 * df['Close'].rolling(window=20).std()
            df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
            df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

            # 5. 动量和波动率特征
            for window in [5, 10, 20]:
                df[f'momentum_{window}'] = df['Close'].pct_change(window)
                df[f'volatility_{window}'] = df['returns'].rolling(window=window).std()
            
            # 6. 填充缺失值
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                df[col] = df[col].interpolate(method='linear')
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
            
            # 7. 生成交易信号
            df['signal'] = 0  # 默认持仓不变
            
            # 将条件分成不同类别，每类条件独立判断
            # 1. 趋势类信号
            trend_signals = {
                'imf_trend': df['IMF_0'] > 0,  # 最高频IMF向上
                'main_trend': df['trend'] > df['trend'].shift(1),  # 总体趋势向上
                'ma_trend': df['Close'] > df['MA20'],  # 价格在中期均线上方
            }
            
            # 2. 技术指标类信号
            tech_signals = {
                'rsi_ok': (df['RSI'] > 30) & (df['RSI'] < 80),  # RSI区间更宽松
                'bb_ok': (df['BB_position'] > 0.1) & (df['BB_position'] < 0.9),  # 布林带位置更宽松
                'macd_ok': df['MACD'] > df['MACD_signal']  # MACD信号
            }
            
            # 3. 动量类信号
            momentum_signals = {
                'short_momentum': df['momentum_5'] > -0.01,  # 短期动量不太差
                'mid_momentum': df['momentum_20'] > -0.02,  # 中期动量不太差
                'vol_ok': df['volatility'] < df['volatility'].rolling(50).mean() * 1.5  # 波动率限制更宽松
            }
            
            # 4. 宏观经济信号
            macro_signals = {
                'cpi_ok': True,
                'gdp_ok': True
            }
            
            if 'CPI_CPI_diff' in df.columns:
                macro_signals['cpi_ok'] = abs(df['CPI_CPI_diff']) < df['CPI_CPI_diff'].std() * 2.5  # 更宽松的CPI限制
            if 'REAL_GDP_REAL_GDP_diff' in df.columns:
                macro_signals['gdp_ok'] = abs(df['REAL_GDP_REAL_GDP_diff']) < df['REAL_GDP_REAL_GDP_diff'].std() * 2.5
            
            # 计算每类信号的得分
            df['trend_score'] = sum(trend_signals.values()).astype(int)
            df['tech_score'] = sum(tech_signals.values()).astype(int)
            df['momentum_score'] = sum(momentum_signals.values()).astype(int)
            df['macro_score'] = sum(macro_signals.values()).astype(int)
            
            # 生成最终信号
            # 多头条件：满足任意2个趋势信号，1个技术信号，1个动量信号
            long_condition = (
                (df['trend_score'] >= 2) & 
                (df['tech_score'] >= 1) & 
                (df['momentum_score'] >= 1) &
                (df['macro_score'] >= 1)
            )
            
            # 空头条件：满足任意2个反向信号
            short_condition = (
                (df['trend_score'] <= 1) & 
                (df['tech_score'] <= 1) & 
                (df['momentum_score'] <= 1) &
                (df['macro_score'] >= 1)  # 宏观经济稳定
            )
            
            # 生成信号
            df.loc[long_condition, 'signal'] = 1
            df.loc[short_condition, 'signal'] = -1
            
            # 8. 信号平滑和风险控制
            # 最小持仓周期
            min_holding_period = 3
            for i in range(min_holding_period, len(df)):
                if df['signal'].iloc[i] != 0 and df['signal'].iloc[i-min_holding_period:i].any():
                    df.loc[df.index[i], 'signal'] = 0
            
            # 止损控制
            stop_loss = 0.015  # 1.5%止损
            for i in range(1, len(df)):
                if df['signal'].iloc[i-1] != 0:  # 如果前一天有持仓
                    returns = df['returns'].iloc[i]
                    if abs(returns) > stop_loss:  # 如果超过止损线
                        df.loc[df.index[i], 'signal'] = 0  # 平仓
            
            # 输出信号统计
            signal_dist = df['signal'].value_counts(normalize=True)
            self.logger.info("\n交易信号分布:")
            for signal in sorted(signal_dist.index):
                self.logger.info(f"信号 {signal}: {signal_dist[signal]*100:.2f}%")
            
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
            imf_features = imf_features.ffill()
            # 后向填充剩余的缺失值
            imf_features = imf_features.bfill()
            
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
                features = features.ffill(0).bffill(0)
            
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
            pairs = ["CNYAUD","CNYEUR", "CNYGBP", "CNYJPY", "CNYUSD"]
            
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

    def _get_countries(self, pair):
        """从货币对获取对应的国家代码"""
        # 货币到国家的映射
        currency_map = {
            'CNY': 'CN',
            'EUR': 'EU', 
            'GBP': 'UK',
            'USD': 'US',
            'JPY': 'JP',
            'AUD': 'AU'  # 添加澳元的映射
        }
        
        # 获取基础货币和报价货币的国家代码
        base_currency = pair[:3]
        quote_currency = pair[3:]
        
        base_country = currency_map.get(base_currency)
        quote_country = currency_map.get(quote_currency)
        
        if not base_country or not quote_country:
            self.logger.error(f"无法映射货币对 {pair} 到对应国家")
            return None, None
        
        return base_country, quote_country

    def _interpolate_macro_data(self, data):
        """对宏观数据进行线性插值处理"""
        try:
            # 确保数据是按日期排序的
            data = data.sort_index()
            
            # 将数据转换为数值类型
            numeric_cols = data.select_dtypes(include=['object']).columns
            for col in numeric_cols:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # 创建完整的日期范围
            date_range = pd.date_range(start=data.index.min(), end=data.index.max(), freq='D')
            
            # 重新索引并进行线性插值
            data_reindexed = data.reindex(date_range)
            data_interpolated = data_reindexed.interpolate(method='linear')
            
            # 处理首尾的缺失值
            data_interpolated = data_interpolated.ffill().bfill()
            
            return data_interpolated
            
        except Exception as e:
            self.logger.error(f"线性插值处理失败: {str(e)}")
            return None

    def _add_macro_features(self, df, pair):
        """添加宏观经济特征"""
        try:
            # 获取货币对的两个国家代码
            country1, country2 = self._get_countries(pair)
            if not country1 or not country2:
                return df
            
            self.logger.info(f"处理国家对: {country1}-{country2}")
            
            # 定义宏观指标及其对应的列名和特征类型
            indicators_config = {
                'CPI': {
                    'columns': ['CPI', 'CPI_YOY'],
                    'features': ['diff']
                },
                'INFLATION': {
                    'columns': ['INFLATION', 'INFLATION_YOY'],
                    'features': ['diff']
                },
                'REAL_GDP': {
                    'columns': ['REAL_GDP'],
                    'features': ['diff']
                },
                'UNEMPLOYMENT': {
                    'columns': ['UNEMPLOYMENT'],
                    'features': ['diff']
                }
            }
            
            # 存储所有国家的宏观数据
            macro_data = {}
            
            # 加载并处理每个国家的宏观数据
            for country in [country1, country2]:
                macro_data[country] = {}
                for indicator, config in indicators_config.items():
                    # 加载数据
                    file_path = os.path.join(self.macro_dir, f'{country}_{indicator}.csv')
                    self.logger.info(f"尝试读取文件: {file_path}")
                    
                    if os.path.exists(file_path):
                        try:
                            # 读取数据并设置索引
                            data = pd.read_csv(file_path)
                            data['date'] = pd.to_datetime(data['date'])
                            data.set_index('date', inplace=True)
                            
                            # 重新采样到日频数据
                            data = data.resample('D').asfreq()
                            
                            # 对每个列进行插值
                            for col in config['columns']:
                                if col in data.columns:
                                    # 使用线性插值填充缺失值
                                    data[col] = data[col].interpolate(method='linear')
                                    # 使用前向填充处理剩余的缺失值
                                    data[col] = data[col].fillna(method='ffill')
                                    # 使用后向填充处理剩余的缺失值
                                    data[col] = data[col].fillna(method='bfill')
                            
                            macro_data[country][indicator] = data
                            self.logger.info(f"成功处理 {country}_{indicator} 数据")
                            
                        except Exception as e:
                            self.logger.error(f"处理{country}_{indicator}数据失败: {str(e)}")
                            continue
                    else:
                        self.logger.warning(f"文件不存在: {file_path}")
            
            # 计算特征
            for indicator, config in indicators_config.items():
                if (indicator in macro_data[country1] and 
                    indicator in macro_data[country2]):
                    
                    self.logger.info(f"开始计算 {indicator} 特征")
                    
                    for col in config['columns']:
                        try:
                            # 获取两个国家的数据
                            data1 = macro_data[country1][indicator][col]
                            data2 = macro_data[country2][indicator][col]
                            
                            # 将数据重新索引到交易数据的日期
                            data1 = data1.reindex(df.index)
                            data2 = data2.reindex(df.index)
                            
                            # 计算差值特征
                            feat_name = f'{indicator}_{col}_diff'
                            df[feat_name] = data1 - data2
                            
                            # 处理缺失值和异常值
                            df[feat_name] = df[feat_name].replace([np.inf, -np.inf], np.nan)
                            # 使用fillna方法替代ffill和bfill
                            df[feat_name] = df[feat_name].fillna(method='ffill')
                            df[feat_name] = df[feat_name].fillna(method='bfill')
                            
                            # 处理异常值
                            self.handle_outliers(df, [feat_name])
                            
                            self.logger.info(f"成功创建特征: {feat_name}")
                            
                        except Exception as e:
                            self.logger.error(f"计算{indicator}_{col}特征失败: {str(e)}")
                            continue
                else:
                    self.logger.warning(f"{indicator} 数据在两个国家中不完整")
            
            # 输出最终生成的特征列表
            macro_features = [col for col in df.columns if any(ind in col for ind in indicators_config.keys())]
            self.logger.info(f"生成的宏观特征列表: {macro_features}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"添加宏观经济特征失败: {str(e)}")
            return df

    def generate_signals(self, df, pair):
        """根据不同货币对生成差异化的交易信号"""
        
        if pair == 'CNYUSD':
            # 1. 计算多个独立的微观信号
            
            # 1.1 基于价格变动的信号
            df['price_signal'] = np.where(
                df['Close'].diff() > 0, 1,
                np.where(df['Close'].diff() < 0, -1, 0)
            )
            
            # 1.2 基于高低点的信号
            df['hl_signal'] = np.where(
                df['High'] > df['High'].shift(1), 1,
                np.where(df['Low'] < df['Low'].shift(1), -1, 0)
            )
            
            # 1.3 基于开盘收盘价差的信号
            df['oc_signal'] = np.where(
                df['Close'] > df['Open'], 1,
                np.where(df['Close'] < df['Open'], -1, 0)
            )
            
            # 1.4 基于极短期动量的信号
            df['mom_signal'] = np.where(
                df['Close'].diff(2).rolling(3).mean() > 0, 1,
                np.where(df['Close'].diff(2).rolling(3).mean() < 0, -1, 0)
            )
            
            # 1.5 基于价格波动的信号
            df['vol_signal'] = np.where(
                (df['High'] - df['Low']) > (df['High'] - df['Low']).shift(1), 1,
                np.where((df['High'] - df['Low']) < (df['High'] - df['Low']).shift(1), -1, 0)
            )
            
            # 2. 计算信号权重
            weights = {
                'price_signal': 0.25,
                'hl_signal': 0.2,
                'oc_signal': 0.2,
                'mom_signal': 0.2,
                'vol_signal': 0.15
            }
            
            # 3. 计算加权组合信号
            df['weighted_signal'] = sum(df[signal] * weight for signal, weight in weights.items())
            
            # 4. 生成最终信号（超级激进版本）
            signal_threshold = 0.1  # 极小的阈值
            
            # 4.1 基础信号
            df['base_signal'] = np.where(
                df['weighted_signal'] > signal_threshold, 1,
                np.where(df['weighted_signal'] < -signal_threshold, -1, 0)
            )
            
            # 4.2 信号增强
            df['signal_count'] = (
                (df['price_signal'] != 0).astype(int) +
                (df['hl_signal'] != 0).astype(int) +
                (df['oc_signal'] != 0).astype(int) +
                (df['mom_signal'] != 0).astype(int) +
                (df['vol_signal'] != 0).astype(int)
            )
            
            # 4.3 最终信号生成（多重条件）
            df['signal'] = 0
            
            # 多头条件：任意两个信号同向且加权信号为正
            long_condition = (
                (df['signal_count'] >= 2) &
                (df['weighted_signal'] > 0) &
                ((df['price_signal'] == 1) | (df['hl_signal'] == 1) | 
                 (df['oc_signal'] == 1) | (df['mom_signal'] == 1) | 
                 (df['vol_signal'] == 1))
            )
            
            # 空头条件：任意两个信号同向且加权信号为负
            short_condition = (
                (df['signal_count'] >= 2) &
                (df['weighted_signal'] < 0) &
                ((df['price_signal'] == -1) | (df['hl_signal'] == -1) | 
                 (df['oc_signal'] == -1) | (df['mom_signal'] == -1) | 
                 (df['vol_signal'] == -1))
            )
            
            # 5. 应用信号
            df.loc[long_condition, 'signal'] = 1
            df.loc[short_condition, 'signal'] = -1
            
            # 6. 信号平滑（可选，但建议保留以减少噪音）
            df['signal'] = df['signal'].rolling(window=2, min_periods=1).mean()
            df['signal'] = np.where(df['signal'] > 0.2, 1,
                                   np.where(df['signal'] < -0.2, -1, 0))
            
            # 7. 清理临时列
            columns_to_drop = [
                'price_signal', 'hl_signal', 'oc_signal', 'mom_signal',
                'vol_signal', 'weighted_signal', 'base_signal',
                'signal_count'
            ]
            df.drop(columns=columns_to_drop, inplace=True)
            
        else:
            # 其他货币对保持原有的信号生成逻辑
            trend_signals = {
                'imf_trend': df['IMF_0'] > 0,  # 最高频IMF向上
                'main_trend': df['trend'] > df['trend'].shift(1),  # 总体趋势向上
                'ma_trend': df['Close'] > df['MA20'],  # 价格在中期均线上方
            }
            
            tech_signals = {
                'rsi_ok': (df['RSI'] > 30) & (df['RSI'] < 80),  # RSI区间更宽松
                'bb_ok': (df['BB_position'] > 0.1) & (df['BB_position'] < 0.9),  # 布林带位置更宽松
                'macd_ok': df['MACD'] > df['MACD_signal']  # MACD信号
            }
            
            momentum_signals = {
                'short_momentum': df['momentum_5'] > -0.01,  # 短期动量不太差
                'mid_momentum': df['momentum_20'] > -0.02,  # 中期动量不太差
                'vol_ok': df['volatility'] < df['volatility'].rolling(50).mean() * 1.5  # 波动率限制更宽松
            }
            
            macro_signals = {
                'cpi_ok': abs(df['CPI_CPI_diff']) < df['CPI_CPI_diff'].std() * 2.5,
                'gdp_ok': abs(df['REAL_GDP_REAL_GDP_diff']) < df['REAL_GDP_REAL_GDP_diff'].std() * 2.5
            }
            
            # 计算各类信号得分
            df['trend_score'] = sum(trend_signals.values())
            df['tech_score'] = sum(tech_signals.values())
            df['momentum_score'] = sum(momentum_signals.values())
            df['macro_score'] = sum(macro_signals.values())
            
            # 常规交易条件
            long_condition = (
                (df['trend_score'] >= 2) &  # 需要2个趋势信号
                (df['tech_score'] >= 1) &   # 需要1个技术信号
                (df['momentum_score'] >= 1) & # 需要1个动量信号
                (df['macro_score'] >= 1)    # 需要1个宏观指标支持
            )
            
            short_condition = (
                (df['trend_score'] <= 1) & 
                (df['tech_score'] <= 1) &
                (df['momentum_score'] <= 1)
            )
        
        # 生成信号
        df.loc[long_condition, 'signal'] = 1
        df.loc[short_condition, 'signal'] = -1
        df.loc[~(long_condition | short_condition), 'signal'] = 0
        
        # 信号平滑处理（避免频繁交易）
        df['signal'] = df['signal'].rolling(window=3, min_periods=1).mean()
        df['signal'] = np.where(df['signal'] > 0.5, 1, 
                               np.where(df['signal'] < -0.5, -1, 0))
        
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