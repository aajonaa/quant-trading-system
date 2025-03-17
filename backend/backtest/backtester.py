import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from datetime import datetime
import seaborn as sns
import itertools

class ForexBacktester:
    def __init__(self, initial_capital=100000):
        self.logger = logging.getLogger(__name__)
        self.current_dir = Path(__file__).parent
        self.signals_dir = self.current_dir.parent / "signals"
        self.output_dir = self.current_dir / "backtest_results"
        self.output_dir.mkdir(exist_ok=True)
        
        self.initial_capital = initial_capital
        self.positions = {}  # 持仓状态
        self.portfolio = {}  # 组合表现
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
    def load_data(self):
        """加载所有货币对的信号数据"""
        try:
            self.pairs_data = {}
            signal_files = list(self.signals_dir.glob("*_backtest.csv"))
            
            if not signal_files:
                self.logger.error("未找到信号文件")
                return False
                
            for file in signal_files:
                pair = file.stem.replace("_signals", "")
                df = pd.read_csv(file)
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                self.pairs_data[pair] = df
                
            self.logger.info(f"成功加载 {len(self.pairs_data)} 个货币对的数据")
            return True
            
        except Exception as e:
            self.logger.error(f"加载数据失败: {str(e)}")
            return False
            
    def run_backtest(self, params=None):
        """执行回测"""
        # 默认参数
        default_params = {
            'commission': 0.0001,      # 手续费
            'stop_loss': 0.02,        # 止损比例
            'take_profit': 0.05,      # 止盈比例
            'max_positions': 3,       # 最大同时持仓数
            'position_size': 0.2,     # 每笔交易资金比例
            'trailing_stop': 0.01,    # 追踪止损比例
            'max_drawdown': 0.15,     # 最大回撤限制
            'min_confidence': 0.6,    # 最小信号置信度
            'vol_threshold': 1.5,     # 波动率阈值
            'trend_filter': True      # 趋势过滤
        }
        
        # 更新参数
        self.params = {**default_params, **(params or {})}
        
        try:
            results = {}
            for pair, df in self.pairs_data.items():
                # 计算波动率
                returns = df['Close'].pct_change()
                volatility = returns.rolling(20).std() * np.sqrt(252)
                
                # 计算趋势
                if self.params['trend_filter']:
                    sma20 = df['Close'].rolling(20).mean()
                    sma50 = df['Close'].rolling(50).mean()
                    trend = (sma20 > sma50).astype(int)
                
                # 初始化结果
                position = 0
                capital = self.initial_capital
                trades = []
                equity_curve = []
                trailing_high = capital
                consecutive_losses = 0
                
                for i in range(1, len(df)):
                    date = df.index[i]
                    signal = df['Signal'].iloc[i]
                    confidence = abs(df['Proba'].iloc[i]) if 'Ensemble_Proba' in df else 1.0
                    price = df['Close'].iloc[i]
                    prev_price = df['Close'].iloc[i-1]
                    
                    # 检查交易条件
                    vol_check = volatility.iloc[i] <= self.params['vol_threshold']
                    conf_check = confidence >= self.params['min_confidence']
                    trend_check = True if not self.params['trend_filter'] else (
                        (signal > 0 and trend.iloc[i] == 1) or 
                        (signal < 0 and trend.iloc[i] == 0)
                    )
                    
                    # 计算回报
                    if position != 0 and prev_price != 0:
                        returns = (price - prev_price) / prev_price * position
                        capital *= (1 + returns - self.params['commission'])
                        
                        # 更新最高点和回撤
                        if capital > trailing_high:
                            trailing_high = capital
                        current_drawdown = (trailing_high - capital) / trailing_high if trailing_high > 0 else 0
                        
                        # 止损检查
                        if returns < -self.params['stop_loss']:
                            position = 0
                            consecutive_losses += 1
                            trades.append({
                                'date': date,
                                'type': 'Stop Loss',
                                'price': price,
                                'capital': capital,
                                'returns': returns,
                                'confidence': confidence
                            })
                        
                        # 检查止盈条件
                        elif returns > self.params['take_profit']:
                            position = 0
                            consecutive_losses = 0
                            trades.append({
                                'date': date,
                                'type': 'Take Profit',
                                'price': price,
                                'capital': capital,
                                'returns': returns
                            })
                        
                        # 检查追踪止损
                        elif current_drawdown > self.params['trailing_stop']:
                            position = 0
                            trades.append({
                                'date': date,
                                'type': 'Trailing Stop',
                                'price': price,
                                'capital': capital,
                                'returns': returns
                            })
                        
                        # 检查最大回撤限制
                        if current_drawdown > self.params['max_drawdown']:
                            break
                    
                    # 开仓逻辑
                    if capital > 0 and vol_check and conf_check and trend_check:
                        position_value = capital * self.params['position_size']
                        if signal > 0 and position <= 0 and consecutive_losses < 3:
                            position = 1
                            trades.append({
                                'date': date,
                                'type': 'Buy',
                                'price': price,
                                'capital': capital,
                                'position_size': position_value,
                                'returns': 0,
                                'confidence': confidence
                            })
                        elif signal < 0 and position >= 0:
                            position = -1
                            trades.append({
                                'date': date,
                                'type': 'Sell',
                                'price': price,
                                'capital': capital,
                                'position_size': position_value,
                                'returns': 0,
                                'confidence': confidence
                            })
                    
                    # 记录权益曲线
                    current_drawdown = (trailing_high - capital) / trailing_high if trailing_high > 0 else 0
                    equity_curve.append({
                        'date': date,
                        'capital': capital,
                        'position': position,
                        'drawdown': current_drawdown
                    })
                
                # 确保至少有两个数据点
                if len(equity_curve) >= 2:
                    # 计算性能指标
                    equity_df = pd.DataFrame(equity_curve)
                    equity_df.set_index('date', inplace=True)
                    returns = equity_df['capital'].pct_change().fillna(0)
                    
                    # 计算更多指标
                    results[pair] = self._calculate_metrics(trades, equity_df, returns)
                else:
                    results[pair] = self._get_default_metrics(pd.DataFrame())
            
            self.backtest_results = results
            return True
            
        except Exception as e:
            self.logger.error(f"回测执行失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def _calculate_metrics(self, trades, equity_df, returns):
        """计算详细的性能指标"""
        try:
            trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
            
            # 确保数据不为空
            if equity_df.empty or len(equity_df) < 2:
                return self._get_default_metrics(trades_df)
            
            # 计算收益率，处理可能的零值
            total_return = (equity_df['capital'].iloc[-1] - self.initial_capital) / self.initial_capital
            
            # 计算风险调整后的收益指标
            returns_std = returns.std()
            neg_returns_std = returns[returns < 0].std()
            
            sharpe_ratio = (
                np.sqrt(252) * returns.mean() / returns_std 
                if returns_std and not np.isnan(returns_std) else 0
            )
            
            sortino_ratio = (
                np.sqrt(252) * returns.mean() / neg_returns_std 
                if neg_returns_std and not np.isnan(neg_returns_std) else 0
            )
            
            # 计算交易相关指标
            if not trades_df.empty and 'returns' in trades_df.columns:
                win_trades = trades_df[trades_df['returns'] > 0]
                loss_trades = trades_df[trades_df['returns'] < 0]
                
                win_rate = len(win_trades) / len(trades_df) if len(trades_df) > 0 else 0
                profit_factor = (
                    abs(win_trades['returns'].sum() / loss_trades['returns'].sum())
                    if not loss_trades.empty and loss_trades['returns'].sum() != 0
                    else np.inf if not win_trades.empty else 0
                )
                
                avg_trade = trades_df['returns'].mean() if not trades_df.empty else 0
                avg_win = win_trades['returns'].mean() if not win_trades.empty else 0
                avg_loss = loss_trades['returns'].mean() if not loss_trades.empty else 0
                
                # 计算最大连续亏损
                if not trades_df.empty:
                    loss_streaks = [
                        sum(1 for _ in g) 
                        for k, g in itertools.groupby(trades_df['returns'] < 0) 
                        if k
                    ]
                    max_consecutive_losses = max(loss_streaks) if loss_streaks else 0
                else:
                    max_consecutive_losses = 0
            else:
                win_rate = profit_factor = avg_trade = avg_win = avg_loss = max_consecutive_losses = 0
            
            # 计算回撤相关指标
            max_drawdown = equity_df['drawdown'].max() if 'drawdown' in equity_df else 0
            recovery_factor = (
                abs(total_return / max_drawdown) 
                if max_drawdown > 0 else np.inf
            )
            calmar_ratio = (
                (returns.mean() * 252) / abs(max_drawdown)
                if max_drawdown > 0 else np.inf
            )
            
            return {
                'trades': trades_df,
                'equity_curve': equity_df,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_trade': avg_trade,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'max_consecutive_losses': max_consecutive_losses,
                'recovery_factor': recovery_factor,
                'calmar_ratio': calmar_ratio
            }
            
        except Exception as e:
            self.logger.error(f"计算性能指标时出错: {str(e)}")
            return self._get_default_metrics(pd.DataFrame())

    def _get_default_metrics(self, trades_df):
        """返回默认的指标值"""
        return {
            'trades': trades_df,
            'equity_curve': pd.DataFrame(),
            'total_return': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'avg_trade': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'max_consecutive_losses': 0,
            'recovery_factor': 0,
            'calmar_ratio': 0
        }
            
    def generate_reports(self):
        """生成回测报告和可视化"""
        try:
            # 创建汇总报告
            summary = []
            for pair, result in self.backtest_results.items():
                summary.append({
                    '货币对': pair,
                    '总收益率': f"{result['total_return']*100:.2f}%",
                    'Sharpe比率': f"{result['sharpe_ratio']:.2f}",
                    'Sortino比率': f"{result['sortino_ratio']:.2f}",
                    '最大回撤': f"{result['max_drawdown']*100:.2f}%",
                    '胜率': f"{result['win_rate']*100:.2f}%",
                    '盈亏比': f"{result['profit_factor']:.2f}",
                    '平均收益': f"{result['avg_trade']*100:.2f}%",
                    '最大连续亏损': f"{result['max_consecutive_losses']}"
                })
            
            # 保存汇总报告
            summary_df = pd.DataFrame(summary)
            summary_df.to_csv(self.output_dir / "backtest_summary.csv", index=False, encoding='utf-8')
            
            # 生成可视化
            for pair, result in self.backtest_results.items():
                self._generate_pair_analysis(pair, result)
            
            self.logger.info(f"回测报告已保存至: {self.output_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"生成报告失败: {str(e)}")
            return False

    def _generate_pair_analysis(self, pair, result):
        """为单个货币对生成详细分析图表"""
        try:
            # 创建子图
            fig = plt.figure(figsize=(15, 12))
            gs = plt.GridSpec(3, 2, figure=fig)
            
            # 1. 权益曲线 (左上)
            ax1 = fig.add_subplot(gs[0, :])
            equity_curve = result['equity_curve']
            ax1.plot(equity_curve.index, equity_curve['capital'], label='权益曲线', color='blue')
            ax1.axhline(y=self.initial_capital, color='gray', linestyle='--', label='初始资金')
            ax1.set_title(f"{pair} 权益曲线")
            ax1.set_ylabel("资金")
            ax1.grid(True)
            ax1.legend()
            
            # 2. 回撤曲线 (中左)
            ax2 = fig.add_subplot(gs[1, 0])
            ax2.fill_between(equity_curve.index, equity_curve['drawdown'] * 100, 0, 
                            color='red', alpha=0.3, label='回撤')
            ax2.set_title("回撤分析")
            ax2.set_ylabel("回撤 (%)")
            ax2.grid(True)
            ax2.legend()
            
            # 3. 交易分布 (中右)
            ax3 = fig.add_subplot(gs[1, 1])
            trades_df = result['trades']
            if not trades_df.empty and 'returns' in trades_df.columns:
                returns_dist = trades_df['returns'] * 100
                sns.histplot(data=returns_dist, bins=30, ax=ax3)
                ax3.set_title("收益分布")
                ax3.set_xlabel("收益率 (%)")
                ax3.set_ylabel("交易次数")
            
            # 4. 交易类型统计 (左下)
            ax4 = fig.add_subplot(gs[2, 0])
            if not trades_df.empty:
                trade_types = trades_df['type'].value_counts()
                trade_types.plot(kind='bar', ax=ax4)
                ax4.set_title("交易类型统计")
                ax4.set_ylabel("次数")
                plt.xticks(rotation=45)
            
            # 5. 关键指标 (右下)
            ax5 = fig.add_subplot(gs[2, 1])
            ax5.axis('off')
            
            # 将指标分组
            metrics = {
                '收益指标': [
                    ('总收益率', f"{result['total_return']*100:.2f}%"),
                    ('年化收益率', f"{result['total_return']*100/len(result['equity_curve'])*252:.2f}%"),
                    ('平均交易收益', f"{result['avg_trade']*100:.2f}%")
                ],
                '风险指标': [
                    ('最大回撤', f"{result['max_drawdown']*100:.2f}%"),
                    ('Sharpe比率', f"{result['sharpe_ratio']:.2f}"),
                    ('Sortino比率', f"{result['sortino_ratio']:.2f}"),
                    ('Calmar比率', f"{result['calmar_ratio']:.2f}")
                ],
                '交易统计': [
                    ('胜率', f"{result['win_rate']*100:.2f}%"),
                    ('盈亏比', f"{result['profit_factor']:.2f}"),
                    ('平均盈利', f"{result['avg_win']*100:.2f}%"),
                    ('平均亏损', f"{result['avg_loss']*100:.2f}%"),
                    ('最大连续亏损', str(result['max_consecutive_losses']))
                ]
            }
            
            # 设置标题
            ax5.text(0.5, 1.0, "回测指标汇总", fontsize=12, fontweight='bold', 
                     ha='center', va='bottom')
            
            # 绘制指标组
            y_pos = 0.85
            for group_name, group_metrics in metrics.items():
                # 绘制组标题
                ax5.text(0.02, y_pos, group_name, fontsize=11, fontweight='bold',
                        color='darkblue')
                y_pos -= 0.05
                
                # 绘制组内指标
                for name, value in group_metrics:
                    # 使用不同的颜色标记正负值
                    if '%' in value and float(value.strip('%')) > 0:
                        value_color = 'green'
                    elif '%' in value and float(value.strip('%')) < 0:
                        value_color = 'red'
                    else:
                        value_color = 'black'
                    
                    ax5.text(0.05, y_pos, f"{name}:", fontsize=10)
                    ax5.text(0.35, y_pos, value, fontsize=10, color=value_color)
                    y_pos -= 0.04
                
                # 组间距
                y_pos -= 0.03
            
            # 添加时间范围信息
            start_date = result['equity_curve'].index[0].strftime('%Y-%m-%d')
            end_date = result['equity_curve'].index[-1].strftime('%Y-%m-%d')
            ax5.text(0.02, 0.02, f"回测区间: {start_date} 至 {end_date}", 
                     fontsize=9, color='gray')
            
            # 调整布局并保存
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{pair}_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"生成{pair}分析图表失败: {str(e)}")

def main():
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建回测器实例
    backtester = ForexBacktester(initial_capital=100000)
    
    # 加载数据
    if not backtester.load_data():
        return
    
    # 执行回测
    if not backtester.run_backtest():
        return
    
    # 生成报告
    backtester.generate_reports()

if __name__ == "__main__":
    main()
