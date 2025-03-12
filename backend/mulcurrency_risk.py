import pandas as pd
import numpy as np
import logging
from pathlib import Path
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from model_analysis import HybridForexModel
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

class MultiCurrencyRiskAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.current_dir = Path(__file__).parent
        self.signals_dir = self.current_dir / "signals"
        self.output_dir = self.current_dir / "mulsignals"
        self.output_dir.mkdir(exist_ok=True)
        self.pairs_data = {}
        self.correlation_matrix = None
        self.risk_signals = None
        self.pair_combinations = []  # 存储货币对组合
        self.combination_signals = {}  # 存储组合风险信号
        self.model = HybridForexModel()
        self.scaler = StandardScaler()
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
    def load_signals(self):
        """加载所有货币对的信号数据"""
        try:
            # 检查signals目录是否存在
            if not self.signals_dir.exists():
                self.logger.error(f"signals目录不存在: {self.signals_dir}")
                return False
            
            # 获取所有信号文件
            signal_files = list(self.signals_dir.glob("*_signals.csv"))
            
            if not signal_files:
                self.logger.error(f"在 {self.signals_dir} 目录下未找到信号文件")
                return False
            
            self.logger.info(f"找到 {len(signal_files)} 个信号文件")
            
            for file in signal_files:
                try:
                    pair = file.stem.replace("_signals", "")
                    self.logger.info(f"正在处理 {pair} 的数据...")
                    
                    # 读取CSV文件
                    df = pd.read_csv(file)
                    self.logger.info(f"成功读取文件，原始形状: {df.shape}")
                    
                    # 检查并处理日期列
                    if 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'])
                        df.set_index('Date', inplace=True)
                        self.logger.info("日期列处理完成")
                    
                    # 检查必要的列
                    required_columns = ['Close', 'Ensemble_Signal']
                    if not all(col in df.columns for col in required_columns):
                        missing_cols = [col for col in required_columns if col not in df.columns]
                        self.logger.error(f"{pair} 缺少必要的列: {missing_cols}")
                        continue
                    
                    # 处理数据中的无效值
                    df = df.replace([np.inf, -np.inf], np.nan)
                    df = df.ffill().bfill()  # 前向和后向填充
                    
                    self.pairs_data[pair] = df
                    self.logger.info(f"成功加载 {pair} 的信号数据，最终形状: {df.shape}")
                    
                except Exception as e:
                    self.logger.error(f"处理 {file} 时出错: {str(e)}")
                    continue
            
            if not self.pairs_data:
                self.logger.error("没有成功加载任何货币对的数据")
                return False
            
            self.logger.info(f"共成功加载了 {len(self.pairs_data)} 个货币对的数据")
            return True
            
        except Exception as e:
            self.logger.error(f"加载信号数据失败: {str(e)}")
            return False
            
    def calculate_correlation(self):
        """计算货币对之间的相关性"""
        try:
            # 对齐所有数据的日期
            aligned_data = {}
            common_dates = None
            
            # 获取所有Close数据并对齐日期
            for pair, df in self.pairs_data.items():
                dates = df.index
                if common_dates is None:
                    common_dates = set(dates)
                else:
                    common_dates = common_dates.intersection(set(dates))
            
            common_dates = sorted(list(common_dates))
            
            # 创建对齐后的收益率数据
            for pair, df in self.pairs_data.items():
                aligned_df = df.loc[common_dates]
                aligned_data[pair] = aligned_df['Close'].pct_change()
            
            # 计算相关系数矩阵
            returns_df = pd.DataFrame(aligned_data, index=common_dates)
            self.correlation_matrix = returns_df.corr()
            
            self.logger.info("相关性矩阵计算完成，数据长度: {}".format(len(common_dates)))
            return True
            
        except Exception as e:
            self.logger.error(f"计算相关性失败: {str(e)}")
            return False
            
    def generate_risk_signals(self):
        """生成多货币对组合风险信号"""
        try:
            if self.correlation_matrix is None:
                self.logger.error("未找到相关性矩阵")
                return False
            
            # 获取所有可能的货币对组合
            pairs = list(self.correlation_matrix.columns)
            self.risk_signals = {}  # 存储风险信号
            
            for pair1, pair2 in combinations(pairs, 2):
                # 计算相关系数
                corr = self.correlation_matrix.loc[pair1, pair2]
                
                # 计算组合风险信号 (-1 到 1)
                risk_signal = np.clip(corr, -1, 1)
                
                combination_name = f"{pair1}_{pair2}"
                self.risk_signals[combination_name] = {
                    'correlation': corr,
                    'risk_signal': risk_signal,
                    'risk_type': "同向风险" if corr > 0 else "对冲风险",
                    'risk_level': abs(risk_signal)
                }
            
            # 生成风险报告
            self.logger.info(f"已生成 {len(self.risk_signals)} 个组合的风险信号")
            return True
            
        except Exception as e:
            self.logger.error(f"生成风险信号失败: {str(e)}")
            return False

    def generate_combination_signals(self):
        """生成两两货币对的风险信号"""
        try:
            if not self.correlation_matrix is not None:
                self.logger.error("未找到相关性矩阵")
                return False
            
            # 创建风险信号列表
            risk_signals = []
            pairs = list(self.correlation_matrix.columns)
            
            # 生成所有两两组合的风险信号
            for pair1, pair2 in combinations(pairs, 2):
                # 获取两个货币对的数据
                df1 = self.pairs_data[pair1]
                df2 = self.pairs_data[pair2]
                
                # 计算相关系数
                corr = self.correlation_matrix.loc[pair1, pair2]
                
                # 计算组合风险指标
                returns1 = df1['Close'].pct_change()
                returns2 = df2['Close'].pct_change()
                
                # 计算组合波动率（年化）
                combined_vol = (returns1 + returns2).std() * np.sqrt(252)
                
                # 计算信号一致性
                signal_agreement = (
                    (df1['Ensemble_Signal'] * df2['Ensemble_Signal']).mean()
                )
                
                # 计算风险得分 (0-100)
                # 使用相关系数、波动率和信号一致性的加权组合
                risk_score = (
                    abs(corr) * 0.4 +  # 相关性权重
                    combined_vol * 0.3 +  # 波动率权重
                    abs(signal_agreement) * 0.3  # 信号一致性权重
                ) * 100
                
                # 确定风险等级和交易建议
                if risk_score >= 30:
                    risk_level = "高风险"
                    trade_advice = "建议对冲"
                elif risk_score >= 20:
                    risk_level = "中风险"
                    trade_advice = "建议减小敞口"
                elif risk_score >= 10:
                    risk_level = "低风险"
                    trade_advice = "建议持有"
                else:
                    risk_level = "极低风险"
                    trade_advice = "可以增加敞口"
                
                # 添加到风险信号列表
                risk_signals.append({
                    '货币对组合': f"{pair1}-{pair2}",
                    '相关系数': round(corr, 4),
                    '组合波动率': round(combined_vol, 4),
                    '信号一致性': round(signal_agreement, 4),
                    '风险得分': round(risk_score, 2),
                    '风险等级': risk_level,
                    '交易建议': trade_advice
                })
            
            # 创建DataFrame并按风险得分排序
            risk_df = pd.DataFrame(risk_signals)
            risk_df = risk_df.sort_values('风险得分', ascending=False)
            
            # 保存到CSV文件
            output_file = self.output_dir / "currency_pair_risks.csv"
            risk_df.to_csv(output_file, index=False, encoding='utf-8')
            self.logger.info(f"已生成货币对风险信号CSV文件: {output_file}")
            
            # 输出风险统计
            self.logger.info("\n=== 风险等级分布 ===")
            level_counts = risk_df['风险等级'].value_counts()
            for level, count in level_counts.items():
                self.logger.info(f"{level}: {count}对")
            
            return True
            
        except Exception as e:
            self.logger.error(f"生成组合风险信号失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def generate_visualizations(self):
        """生成风险可视化"""
        try:
            # 1. 创建相关性热力图
            plt.figure(figsize=(12, 8))
            sns.heatmap(
                self.correlation_matrix,
                annot=True,
                cmap='RdYlBu_r',
                center=0,
                vmin=-1,
                vmax=1,
                fmt='.2f'
            )
            plt.title('货币对相关性热力图')
            plt.tight_layout()
            
            # 保存热力图
            correlation_path = self.output_dir / "correlation_heatmap.png"
            plt.savefig(correlation_path)
            plt.close()
            
            # 2. 创建风险指标热力图
            risk_data = pd.DataFrame(index=self.pairs_data.keys())
            
            # 计算风险指标
            for pair, df in self.pairs_data.items():
                returns = df['Close'].pct_change()
                risk_data.loc[pair, '波动率'] = returns.std() * np.sqrt(252)  # 年化波动率
                risk_data.loc[pair, '偏度'] = returns.skew()
                risk_data.loc[pair, '峰度'] = returns.kurtosis()
                risk_data.loc[pair, '最大回撤'] = (df['Close'] / df['Close'].expanding().max() - 1).min()
                risk_data.loc[pair, '信号强度'] = abs(df['Ensemble_Signal']).mean()
            
            # 标准化风险指标
            risk_data = (risk_data - risk_data.mean()) / risk_data.std()
            
            # 创建风险热力图
            plt.figure(figsize=(12, 8))
            sns.heatmap(
                risk_data,
                annot=True,
                cmap='RdYlBu_r',
                center=0,
                fmt='.2f'
            )
            plt.title('货币对风险指标热力图')
            plt.tight_layout()
            
            # 保存风险热力图
            risk_path = self.output_dir / "risk_heatmap.png"
            plt.savefig(risk_path)
            plt.close()
            
            # 3. 创建风险时序图
            plt.figure(figsize=(15, 8))
            for pair in self.pairs_data.keys():
                df = self.pairs_data[pair]
                risk = df['Close'].pct_change().rolling(20).std() * np.sqrt(252)
                plt.plot(df.index, risk, label=pair, alpha=0.7)
            
            plt.title('货币对风险时序变化')
            plt.xlabel('日期')
            plt.ylabel('年化波动率')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            # 保存风险时序图
            timeseries_path = self.output_dir / "risk_timeseries.png"
            plt.savefig(timeseries_path)
            plt.close()
            
            self.logger.info(f"已生成可视化图表：")
            self.logger.info(f"1. 相关性热力图: {correlation_path}")
            self.logger.info(f"2. 风险指标热力图: {risk_path}")
            self.logger.info(f"3. 风险时序图: {timeseries_path}")
            
            return {
                'correlation_heatmap': str(correlation_path),
                'risk_heatmap': str(risk_path),
                'risk_timeseries': str(timeseries_path)
            }
            
        except Exception as e:
            self.logger.error(f"生成可视化时出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def analyze_currency_risks(self, data_dict):
        """分析多个货币对的风险，使用ensemble_signal"""
        try:
            results = {}
            for pair, data in data_dict.items():
                self.logger.info(f"\n分析 {pair}...")
                
                # 数据预处理
                data = data.replace([np.inf, -np.inf], np.nan)
                data = data.ffill()  # 使用前值填充
                
                # 使用 Ensemble_Signal 作为基础信号
                if 'Ensemble_Signal' in data.columns:
                    signals = data['Ensemble_Signal']
                    
                    # 计算风险指标
                    risk_metrics = {
                        'mean_signal': signals.mean(),  # 平均信号强度
                        'signal_volatility': signals.std(),  # 信号波动性
                        'signal_skew': signals.skew(),  # 信号偏度
                        'signal_ratios': {  # 各类信号的比例
                            'up': (signals == 1).mean(),
                            'down': (signals == -1).mean(),
                            'neutral': (signals == 0).mean()
                        }
                    }
                    
                    # 计算风险评分
                    risk_score = self._calculate_signal_risk_score(signals)
                    risk_metrics['risk_score'] = risk_score
                    
                    results[pair] = {
                        'signals': signals,
                        'metrics': risk_metrics
                    }
                    
                    # 输出风险评估结果
                    self._print_signal_assessment(pair, risk_metrics)
                else:
                    self.logger.error(f"{pair} 缺少 Ensemble_Signal 列")
                    continue
            
            return results
            
        except Exception as e:
            self.logger.error(f"风险分析失败: {str(e)}")
            return None

    def _calculate_signal_risk_score(self, signals):
        """基于信号计算风险评分"""
        try:
            # 信号稳定性分数 (0-40分)
            stability = 1 - abs(signals).std()
            stability_score = np.clip(stability * 40, 0, 40)
            
            # 信号一致性分数 (0-30分)
            consistency = 1 - abs(np.diff(signals)).mean()
            consistency_score = consistency * 30
            
            # 信号强度分数 (0-30分)
            strength = abs(signals.mean())
            strength_score = strength * 30
            
            # 综合评分
            total_score = stability_score + consistency_score + strength_score
            
            return np.clip(total_score, 0, 100)
            
        except Exception as e:
            self.logger.error(f"风险评分计算失败: {str(e)}")
            return 0

    def _print_signal_assessment(self, pair, metrics):
        """输出信号风险评估结果"""
        try:
            self.logger.info(f"\n=== {pair} 风险评估报告 ===")
            self.logger.info(f"平均信号强度: {metrics['mean_signal']:.4f}")
            self.logger.info(f"信号波动性: {metrics['signal_volatility']:.4f}")
            self.logger.info(f"信号偏度: {metrics['signal_skew']:.4f}")
            self.logger.info("\n信号分布:")
            for direction, ratio in metrics['signal_ratios'].items():
                self.logger.info(f"{direction}: {ratio:.4f}")
            self.logger.info(f"\n风险评分: {metrics['risk_score']:.2f}/100")
            
            # 风险等级评估
            risk_level = self._get_risk_level(metrics['risk_score'])
            self.logger.info(f"风险等级: {risk_level}")
            
        except Exception as e:
            self.logger.error(f"风险评估输出失败: {str(e)}")

    def _get_risk_level(self, risk_score):
        """根据风险评分确定风险等级"""
        if risk_score >= 80:
            return "低风险"
        elif risk_score >= 60:
            return "中等风险"
        elif risk_score >= 40:
            return "高风险"
        else:
            return "极高风险"

    def calculate_risk_signals(self, currency_data):
        """
        计算多货币风险信号
        
        计算逻辑：
        1. 计算每个货币对的波动率
        2. 计算相关性矩阵
        3. 综合评估系统性风险
        4. 生成风险信号
        """
        try:
            # 提取所有货币对的收益率
            returns_dict = {}
            for pair, df in currency_data.items():
                returns_dict[pair] = df['Close'].pct_change()
            
            returns_df = pd.DataFrame(returns_dict)
            
            # 计算20日滚动波动率
            volatilities = returns_df.rolling(window=20).std() * np.sqrt(252)  # 年化
            
            # 计算20日滚动相关性矩阵
            correlations = returns_df.rolling(window=20).corr()
            
            # 计算系统性风险指标
            risk_signals = pd.DataFrame(index=returns_df.index)
            
            # 1. 平均波动率
            risk_signals['avg_vol'] = volatilities.mean(axis=1)
            
            # 2. 平均相关性
            daily_corr = []
            for date in returns_df.index:
                if date in correlations.index.levels[0]:
                    corr_matrix = correlations.loc[date]
                    # 提取上三角矩阵的相关系数（排除对角线）
                    upper_triangle = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
                    daily_corr.append(np.mean(upper_triangle))
                else:
                    daily_corr.append(np.nan)
            
            risk_signals['avg_corr'] = daily_corr
            
            # 3. 计算综合风险指标
            risk_signals['risk_indicator'] = (
                (risk_signals['avg_vol'] - risk_signals['avg_vol'].rolling(60).mean()) / risk_signals['avg_vol'].rolling(60).std() +
                (risk_signals['avg_corr'] - risk_signals['avg_corr'].rolling(60).mean()) / risk_signals['avg_corr'].rolling(60).std()
            )
            
            # 4. 生成风险信号
            risk_signals['risk_level'] = pd.qcut(
                risk_signals['risk_indicator'].fillna(method='ffill'),
                q=3,
                labels=['低风险', '中等风险', '高风险']
            )
            
            return risk_signals, correlations
            
        except Exception as e:
            self.logger.error(f"计算风险信号时出错: {str(e)}")
            return None, None

    def plot_risk_heatmap(self, currency_data, risk_signals, correlations, output_path=None):
        """生成风险热力图"""
        try:
            # 创建图形
            fig = plt.figure(figsize=(15, 10))
            
            # 1. 相关性热力图
            plt.subplot(2, 1, 1)
            latest_corr = correlations.iloc[-1]
            latest_corr = latest_corr.unstack()
            
            sns.heatmap(latest_corr, 
                       annot=True, 
                       cmap='RdYlBu_r',
                       center=0,
                       vmin=-1,
                       vmax=1,
                       fmt='.2f')
            plt.title('货币对相关性热力图')
            
            # 2. 风险指标热力图
            plt.subplot(2, 1, 2)
            risk_data = pd.DataFrame(index=currency_data.keys())
            
            # 计算每个货币对的风险指标
            for pair, df in currency_data.items():
                risk_data.loc[pair, '波动率'] = df['Close'].pct_change().std() * np.sqrt(252)
                risk_data.loc[pair, '偏度'] = df['Close'].pct_change().skew()
                risk_data.loc[pair, '峰度'] = df['Close'].pct_change().kurtosis()
                risk_data.loc[pair, '最大回撤'] = (df['Close'] / df['Close'].expanding().max() - 1).min()
            
            # 标准化风险指标
            risk_data = (risk_data - risk_data.mean()) / risk_data.std()
            
            sns.heatmap(risk_data,
                       annot=True,
                       cmap='RdYlBu_r',
                       center=0,
                       fmt='.2f')
            plt.title('货币对风险指标热力图')
            
            plt.tight_layout()
            
            # 保存图形
            if output_path:
                plt.savefig(output_path)
                self.logger.info(f"热力图已保存至: {output_path}")
            
            plt.close()
            
            return risk_data
            
        except Exception as e:
            self.logger.error(f"生成热力图时出错: {str(e)}")
            return None

    def analyze_risk(self, currency_data):
        """分析多货币风险"""
        try:
            # 1. 计算每个货币对的收益率
            for pair, df in currency_data.items():
                df['returns'] = df['Close'].pct_change()
            
            # 2. 计算风险信号
            risk_signals, correlations = self.calculate_risk_signals(currency_data)
            if risk_signals is None:
                return None
            
            # 3. 生成热力图
            output_path = self.output_dir / f"risk_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            risk_data = self.plot_risk_heatmap(currency_data, risk_signals, correlations, output_path)
            
            # 4. 生成风险报告
            latest_risk = risk_signals.iloc[-1]
            report = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'risk_level': latest_risk['risk_level'],
                'avg_volatility': latest_risk['avg_vol'],
                'avg_correlation': latest_risk['avg_corr'],
                'risk_indicator': latest_risk['risk_indicator'],
                'risk_metrics': risk_data.to_dict() if risk_data is not None else None
            }
            
            # 5. 添加详细的风险指标
            for pair, df in currency_data.items():
                report[f'{pair}_metrics'] = {
                    'volatility': df['returns'].std() * np.sqrt(252),  # 年化波动率
                    'skewness': df['returns'].skew(),  # 偏度
                    'kurtosis': df['returns'].kurtosis(),  # 峰度
                    'max_drawdown': (df['Close'] / df['Close'].expanding().max() - 1).min(),  # 最大回撤
                    'signal_strength': abs(df['Ensemble_Signal']).mean()  # 信号强度
                }
            
            return report
            
        except Exception as e:
            self.logger.error(f"分析风险时出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None


def main():
    try:
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 创建分析器实例
        analyzer = MultiCurrencyRiskAnalyzer()
        
        # 加载信号数据
        logging.info("开始加载信号数据...")
        if not analyzer.load_signals():
            logging.error("加载信号数据失败，程序终止")
            return
            
        # 计算相关性
        logging.info("开始计算相关性...")
        if not analyzer.calculate_correlation():
            logging.error("计算相关性失败，程序终止")
            return
            
        # 生成风险信号
        logging.info("开始生成风险信号...")
        if not analyzer.generate_risk_signals():
            logging.error("生成风险信号失败，程序终止")
            return
            
        # 生成组合风险信号
        logging.info("开始生成组合风险信号...")
        if not analyzer.generate_combination_signals():
            logging.error("生成组合风险信号失败，程序终止")
            return
            
        # 生成可视化
        logging.info("开始生成可视化...")
        visualizations = analyzer.generate_visualizations()
        if visualizations:
            logging.info("多货币风险分析和可视化完成，数据已保存到 mulsignals 目录")
        
        # 执行风险分析
        logging.info("开始执行风险分析...")
        results = analyzer.analyze_risk(analyzer.pairs_data)
        
        # 输出风险分析总结
        if results:
            logging.info("\n=== 多货币风险分析总结 ===")
            logging.info(f"当前风险等级: {results['risk_level']}")
            logging.info(f"平均波动率: {results['avg_volatility']:.4f}")
            logging.info(f"平均相关性: {results['avg_correlation']:.4f}")
            logging.info(f"风险指标: {results['risk_indicator']:.4f}")
            
            # 输出每个货币对的详细指标
            logging.info("\n各货币对风险指标:")
            for pair in analyzer.pairs_data.keys():
                metrics = results[f'{pair}_metrics']
                logging.info(f"\n{pair}:")
                logging.info(f"  年化波动率: {metrics['volatility']:.4f}")
                logging.info(f"  偏度: {metrics['skewness']:.4f}")
                logging.info(f"  峰度: {metrics['kurtosis']:.4f}")
                logging.info(f"  最大回撤: {metrics['max_drawdown']:.4f}")
                logging.info(f"  信号强度: {metrics['signal_strength']:.4f}")
        
        logging.info("所有分析完成")
        
    except Exception as e:
        logging.error(f"程序执行过程中出错: {str(e)}")

if __name__ == "__main__":
    main()
