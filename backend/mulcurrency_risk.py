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
        
    def load_signals(self):
        """加载所有货币对的信号数据"""
        try:
            # 获取所有信号文件
            signal_files = list(self.signals_dir.glob("*_signals.csv"))
            
            for file in signal_files:
                pair = file.stem.replace("_signals", "")
                df = pd.read_csv(file, index_col='Date', parse_dates=True)
                self.pairs_data[pair] = df
                
            self.logger.info(f"已加载 {len(self.pairs_data)} 个货币对的数据")
            return True
            
        except Exception as e:
            self.logger.error(f"加载信号数据失败: {str(e)}")
            return False
            
    def calculate_correlation(self):
        """计算货币对之间的相关性"""
        try:
            # 创建收益率DataFrame
            returns_dict = {}
            for pair, df in self.pairs_data.items():
                returns_dict[pair] = df['Close'].pct_change()
                
            # 计算相关系数矩阵
            returns_df = pd.DataFrame(returns_dict)
            self.correlation_matrix = returns_df.corr()
            
            return True
            
        except Exception as e:
            self.logger.error(f"计算相关性失败: {str(e)}")
            return False
            
    def generate_risk_signals(self):
        """生成综合风险信号"""
        try:
            # 创建信号DataFrame
            signals_dict = {}
            for pair, df in self.pairs_data.items():
                signals_dict[pair] = df['Ensemble_Signal']
                
            signals_df = pd.DataFrame(signals_dict)
            
            # 计算加权风险信号
            weighted_signals = pd.DataFrame(index=signals_df.index)
            
            for pair in signals_df.columns:
                # 获取与当前货币对相关的其他货币对的相关系数
                correlations = self.correlation_matrix[pair].abs()
                # 归一化相关系数作为权重
                weights = correlations / correlations.sum()
                
                # 计算加权信号
                weighted_signal = 0
                for other_pair, weight in weights.items():
                    weighted_signal += signals_df[other_pair] * weight
                
                weighted_signals[pair] = weighted_signal
                
            self.risk_signals = weighted_signals
            
            # 保存风险信号到CSV
            self.risk_signals.to_csv(self.output_dir / "risk_signals.csv")
            
            # 生成风险评估报告
            self._generate_risk_report()
            
            return True
            
        except Exception as e:
            self.logger.error(f"生成风险信号失败: {str(e)}")
            return False
            
    def _generate_risk_report(self):
        """生成风险评估报告"""
        try:
            report = {}
            
            # 计算每个货币对的风险指标
            for pair in self.risk_signals.columns:
                signals = self.risk_signals[pair]
                
                report[pair] = {
                    'risk_level': abs(signals.mean()),  # 平均风险水平
                    'risk_volatility': signals.std(),   # 风险波动性
                    'max_risk': abs(signals.max()),     # 最大风险
                    'risk_direction': np.sign(signals.mean())  # 风险方向
                }
                
            self.risk_report = report
            
            # 保存风险报告到CSV
            report_df = pd.DataFrame(report).T
            report_df.to_csv(self.output_dir / "risk_report.csv")
            
            return True
            
        except Exception as e:
            self.logger.error(f"生成风险报告失败: {str(e)}")
            return False
            
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
        """为每个货币对组合生成综合风险信号"""
        try:
            for comb in self.pair_combinations:
                pairs = comb[:-1]  # 最后一个元素是相关性
                
                # 获取这些货币对的信号
                signals = pd.DataFrame()
                for pair in pairs:
                    signals[pair] = self.risk_signals[pair]
                
                # 计算组合风险信号
                # 1. 基于相关性的加权平均
                weights = np.array([abs(self.correlation_matrix.loc[pairs[0], pair]) 
                                  for pair in pairs])
                weights = weights / weights.sum()
                
                # 2. 计算加权风险信号
                combined_signal = np.zeros(len(signals))
                for i, pair in enumerate(pairs):
                    combined_signal += weights[i] * signals[pair].values
                
                # 保存组合信号
                comb_name = "_".join(pairs)
                self.combination_signals[comb_name] = pd.Series(
                    combined_signal, 
                    index=self.risk_signals.index
                )
            
            # 保存组合风险信号到CSV
            comb_signals_df = pd.DataFrame(self.combination_signals)
            comb_signals_df.to_csv(self.output_dir / "combination_signals.csv")
            
            return True
            
        except Exception as e:
            self.logger.error(f"生成组合风险信号失败: {str(e)}")
            return False

    def generate_visualizations(self):
        """生成可视化数据"""
        try:
            # 1. 相关性热力图
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
                yaxis_title='货币对'
            )
            
            # 2. 风险信号时间序列图
            risk_fig = make_subplots(rows=len(self.pairs_data), cols=1,
                                   subplot_titles=list(self.pairs_data.keys()))
            
            for i, pair in enumerate(self.pairs_data.keys(), 1):
                risk_fig.add_trace(
                    go.Scatter(
                        x=self.risk_signals.index,
                        y=self.risk_signals[pair],
                        name=pair,
                        mode='lines'
                    ),
                    row=i, col=1
                )
                
            risk_fig.update_layout(
                height=300*len(self.pairs_data),
                title_text="多货币对风险信号",
                showlegend=True
            )
            
            # 3. 风险评估仪表盘
            gauge_fig = make_subplots(
                rows=1, cols=len(self.pairs_data),
                subplot_titles=list(self.pairs_data.keys()),
                specs=[[{'type': 'indicator'}]*len(self.pairs_data)]
            )
            
            for i, pair in enumerate(self.pairs_data.keys(), 1):
                gauge_fig.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=abs(self.risk_report[pair]['risk_level']),
                        title={'text': f"{pair} 风险水平"},
                        gauge={'axis': {'range': [0, 1]},
                               'bar': {'color': "red" if self.risk_report[pair]['risk_direction'] < 0 else "green"}}
                    ),
                    row=1, col=i
                )
                
            gauge_fig.update_layout(
                height=400,
                title_text="货币对风险仪表盘"
            )
            
            # 4. 货币对组合风险信号图
            n_combs = len(self.combination_signals)
            comb_fig = make_subplots(
                rows=n_combs, 
                cols=1,
                subplot_titles=list(self.combination_signals.keys()),
                vertical_spacing=0.05
            )
            
            for i, (comb_name, signal) in enumerate(self.combination_signals.items(), 1):
                comb_fig.add_trace(
                    go.Scatter(
                        x=signal.index,
                        y=signal.values,
                        name=comb_name,
                        mode='lines'
                    ),
                    row=i, col=1
                )
            
            comb_fig.update_layout(
                height=250*n_combs,
                title_text="货币对组合风险信号",
                showlegend=True
            )
            
            # 保存图片（使用kaleido后端）
            correlation_fig.write_image(str(self.output_dir / "correlation.png"))
            risk_fig.write_image(str(self.output_dir / "risk_signals.png"))
            gauge_fig.write_image(str(self.output_dir / "risk_gauge.png"))
            comb_fig.write_image(str(self.output_dir / "combination_signals.png"))
            
            # 将图形转换为JSON
            visualizations = {
                'correlation': correlation_fig.to_json(),
                'risk_signals': risk_fig.to_json(),
                'risk_gauge': gauge_fig.to_json(),
                'combination_signals': comb_fig.to_json()
            }
            
            # 保存JSON
            with open(self.output_dir / "visualizations.json", "w") as f:
                json.dump(visualizations, f)
            
            return visualizations
            
        except Exception as e:
            self.logger.error(f"生成可视化失败: {str(e)}")
            return None

def main():
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建分析器实例
    analyzer = MultiCurrencyRiskAnalyzer()
    
    # 加载数据
    if not analyzer.load_signals():
        return
        
    # 计算相关性
    if not analyzer.calculate_correlation():
        return
        
    # 生成风险信号
    if not analyzer.generate_risk_signals():
        return
        
    # 生成货币对组合
    if not analyzer.generate_pair_combinations(correlation_threshold=0.5):
        return
        
    # 生成组合风险信号
    if not analyzer.generate_combination_signals():
        return
    
    # 生成可视化
    visualizations = analyzer.generate_visualizations()
    if visualizations:
        logging.info("多货币风险分析完成，数据已保存到 mulsignals 目录")

if __name__ == "__main__":
    main()
