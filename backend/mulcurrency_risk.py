import pandas as pd
import numpy as np
import logging
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import plotly.io as pio
from itertools import combinations
import kaleido  # 添加这行以支持图片导出
from sklearn.preprocessing import StandardScaler
from model_analysis import HybridForexModel

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
            self._generate_risk_report(self.risk_signals)
            
            self.logger.info(f"已生成 {len(self.risk_signals)} 个组合的风险信号")
            return True
            
        except Exception as e:
            self.logger.error(f"生成风险信号失败: {str(e)}")
            return False

    def _generate_risk_report(self, risk_signals):
        """生成优化后的风险报告"""
        try:
            report_data = []
            for comb_name, metrics in risk_signals.items():
                report_data.append({
                    '货币对组合': comb_name,
                    '相关系数': metrics['correlation'],
                    '风险信号': metrics['risk_signal'],
                    '风险类型': metrics['risk_type'],
                    '风险强度': metrics['risk_level']
                })
            
            # 创建报告DataFrame
            report_df = pd.DataFrame(report_data)
            
            # 按风险强度排序
            report_df = report_df.sort_values('风险强度', ascending=False)
            
            # 保存报告
            report_df.to_csv(self.output_dir / "risk_report.csv", index=False, encoding='utf-8')
            
            # 生成Markdown格式的报告
            md_content = [
                "# 多货币对风险分析报告\n",
                "## 风险信号说明",
                "- 风险信号范围: [-1, 1]",
                "- 正值表示同向风险，负值表示对冲风险",
                "- 绝对值越大表示关联性越强\n",
                "## 主要风险组合\n"
            ]
            
            # 添加高风险组合
            high_risk = report_df[report_df['风险强度'] >= 0.7]
            if not high_risk.empty:
                md_content.append("### 高度关联组合")
                for _, row in high_risk.iterrows():
                    md_content.append(
                        f"- {row['货币对组合']}: {row['风险类型']} "
                        f"(信号强度: {row['风险信号']:.3f})"
                    )
            
            # 添加对冲组合
            hedge_pairs = report_df[report_df['风险信号'] < -0.5]
            if not hedge_pairs.empty:
                md_content.append("\n### 潜在对冲组合")
                for _, row in hedge_pairs.iterrows():
                    md_content.append(
                        f"- {row['货币对组合']}: 对冲效果 "
                        f"{abs(row['风险信号'])*100:.1f}%"
                    )
            
            # 保存Markdown报告
            with open(self.output_dir / "risk_analysis.md", "w", encoding='utf-8') as f:
                f.write("\n".join(md_content))
            
            self.logger.info("风险报告生成完成")
            
        except Exception as e:
            self.logger.error(f"生成风险报告失败: {str(e)}")

    def generate_pair_combinations(self, correlation_threshold=0.5):
        """生成高相关性的货币对组合"""
        try:
            pairs = list(self.correlation_matrix.columns)
            self.pair_combinations = []
            
            # 只生成两两组合
            for pair1, pair2 in combinations(pairs, 2):
                corr = abs(self.correlation_matrix.loc[pair1, pair2])
                if corr >= correlation_threshold:
                    self.pair_combinations.append((pair1, pair2, corr))
            
            # 按相关性排序
            self.pair_combinations.sort(key=lambda x: x[-1], reverse=True)
            
            # 生成组合说明文档
            self._generate_combination_report()
            
            return True
            
        except Exception as e:
            self.logger.error(f"生成货币对组合失败: {str(e)}")
            return False
            
    def _generate_combination_report(self):
        """生成组合分析报告"""
        try:
            report_content = [
                "# 多货币对风险分析报告\n",
                "## 1. 数据文件说明\n",
                "- `risk_signals.csv`: 各货币对的风险信号时间序列数据",
                "- `risk_report.csv`: 各货币对的风险指标统计",
                "- `combination_signals.csv`: 高相关性货币对组合的风险信号",
                "- `visualizations.json`: 交互式可视化数据\n",
                "## 2. 可视化图表说明\n",
                "- `correlation.png`: 货币对之间的相关性热力图",
                "- `risk_signals.png`: 各货币对的风险信号时间序列图",
                "- `risk_gauge.png`: 各货币对当前风险水平仪表盘",
                "- `combination_signals.png`: 高相关性货币对组合的风险信号图\n",
                "## 3. 高相关性货币对组合\n"
            ]
            
            # 添加货币对组合信息
            for pair1, pair2, corr in self.pair_combinations:
                report_content.append(
                    f"- {pair1} - {pair2}: 相关系数 = {corr:.4f}"
                )
            
            report_content.extend([
                "\n## 4. 风险指标说明\n",
                "- risk_level: 平均风险水平（0-1之间，越大风险越高）",
                "- risk_volatility: 风险波动性（风险水平的标准差）",
                "- max_risk: 最大风险值",
                "- risk_direction: 风险方向（正值表示上涨风险，负值表示下跌风险）\n",
                "## 5. 组合风险信号说明\n",
                "组合风险信号通过以下步骤生成：",
                "1. 基于收益率计算货币对之间的相关性",
                "2. 筛选相关系数大于阈值(0.5)的货币对组合",
                "3. 使用相关系数作为权重计算组合风险信号",
                "4. 信号值越大表示组合风险越高\n",
                "## 6. 使用建议\n",
                "1. 关注高相关性货币对的协同变动",
                "2. 结合风险方向和水平进行交易决策",
                "3. 注意风险波动性较大的货币对",
                "4. 优先考虑相关性高的货币对组合"
            ])
            
            # 保存报告
            with open(self.output_dir / "analysis_report.md", "w", encoding='utf-8') as f:
                f.write("\n".join(report_content))
            
        except Exception as e:
            self.logger.error(f"生成分析报告失败: {str(e)}")
            
    def generate_combination_signals(self):
        """生成组合风险信号的时间序列"""
        try:
            if not self.pairs_data:
                self.logger.error("未找到货币对数据")
                return False
            
            # 获取共同的日期索引
            common_dates = None
            for df in self.pairs_data.values():
                dates = set(df.index)
                if common_dates is None:
                    common_dates = dates
                else:
                    common_dates = common_dates.intersection(dates)
            
            common_dates = sorted(list(common_dates))
            
            # 创建组合信号DataFrame
            combination_signals = pd.DataFrame(index=common_dates)
            
            # 对每个货币对组合生成时间序列信号
            for pair1, pair2 in combinations(self.pairs_data.keys(), 2):
                # 获取两个货币对的收益率
                returns1 = self.pairs_data[pair1].loc[common_dates, 'Close'].pct_change()
                returns2 = self.pairs_data[pair2].loc[common_dates, 'Close'].pct_change()
                
                # 计算动态相关系数（使用60天滚动窗口）
                rolling_corr = returns1.rolling(window=60).corr(returns2)
                
                # 生成组合名称
                combination_name = f"{pair1}_{pair2}"
                
                # 保存到组合信号DataFrame
                combination_signals[combination_name] = rolling_corr
            
            self.combination_signals = combination_signals
            
            # 保存组合信号到CSV
            self.combination_signals.to_csv(self.output_dir / "combination_signals.csv")
            self.logger.info(f"已生成 {len(self.combination_signals.columns)} 个组合的时间序列信号")
            
            return True
            
        except Exception as e:
            self.logger.error(f"生成组合风险信号失败: {str(e)}")
            return False

    def generate_visualizations(self):
        """生成可视化数据"""
        try:
            # 1. 相关性热力图
            if self.correlation_matrix is not None:
                correlation_fig = go.Figure(data=go.Heatmap(
                    z=self.correlation_matrix.values,
                    x=self.correlation_matrix.columns,
                    y=self.correlation_matrix.index,
                    colorscale='RdBu',
                    zmin=-1, zmax=1
                ))
                correlation_fig.update_layout(
                    title='货币对相关性热力图',
                    xaxis_title='货币对',
                    yaxis_title='货币对',
                    width=800,
                    height=600
                )
                
                # 保存相关性热力图
                correlation_fig.write_image(str(self.output_dir / "correlation.png"))
                self.logger.info("相关性热力图已保存")
                
                # 2. 组合信号时间序列图
                if self.combination_signals is not None:
                    comb_fig = go.Figure()
                    
                    for column in self.combination_signals.columns:
                        comb_fig.add_trace(
                            go.Scatter(
                                x=self.combination_signals.index,
                                y=self.combination_signals[column],
                                name=column,
                                mode='lines'
                            )
                        )
                        
                    comb_fig.update_layout(
                        title='货币对组合风险信号（动态相关性）',
                        xaxis_title='日期',
                        yaxis_title='相关系数',
                        width=1200,
                        height=800,
                        showlegend=True
                    )
                    
                    # 保存组合信号图
                    comb_fig.write_image(str(self.output_dir / "combination_signals.png"))
                    self.logger.info("组合信号图已保存")
                    
                    return {
                        'correlation': correlation_fig.to_json(),
                        'combination_signals': comb_fig.to_json()
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"生成可视化失败: {str(e)}")
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
            
        # 生成货币对组合
        logging.info("开始生成货币对组合...")
        if not analyzer.generate_pair_combinations(correlation_threshold=0.5):
            logging.error("生成货币对组合失败，程序终止")
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
        results = analyzer.analyze_currency_risks(analyzer.pairs_data)
        
        # 输出风险分析总结
        if results:
            logging.info("\n=== 多货币风险分析总结 ===")
            for pair, result in results.items():
                logging.info(f"{pair} 风险评分: {result['metrics']['risk_score']:.2f}/100")
        
        logging.info("所有分析完成")
        
    except Exception as e:
        logging.error(f"程序执行过程中出错: {str(e)}")

if __name__ == "__main__":
    main()
