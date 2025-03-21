import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import seaborn as sns
import matplotlib.gridspec as gridspec

class ForexBacktester:
    def __init__(self):
        self.signals_dir = Path(__file__).parent.parent / 'signals'
        self.initial_capital = 100000  # 设置初始资金为10万
        self.pairs_data = {}
        self.logger = logging.getLogger(__name__)
        # 初始化时就加载数据
        self.load_data()

    def load_data(self):
        """加载所有货币对的信号数据"""
        try:
            signal_files = list(self.signals_dir.glob("*_signals.csv"))
            
            if not signal_files:
                self.logger.error("未找到信号文件")
                return False
                
            for file in signal_files:
                pair = file.stem.replace("_signals", "")
                self.logger.info(f"加载 {pair} 的信号数据...")
                
                # 读取信号文件
                df = pd.read_csv(file)
                df['Date'] = pd.to_datetime(df['Date'])
                
                # 确保必要的列存在
                required_cols = ['Date', 'Price', 'Signal']
                if not all(col in df.columns for col in required_cols):
                    self.logger.error(f"{pair} 信号文件缺少必要列")
                    continue
                
                # 添加到数据字典
                self.pairs_data[pair] = df
                self.logger.info(f"成功加载 {pair} 数据，共 {len(df)} 条记录")
            
            self.logger.info(f"成功加载 {len(self.pairs_data)} 个货币对的数据")
            return True
            
        except Exception as e:
            self.logger.error(f"加载数据失败: {str(e)}")
            return False

    def run_backtest(self, currency_pair=None, start_date=None, end_date=None):
        """执行回测"""
        try:
            if currency_pair not in self.pairs_data:
                raise Exception(f"找不到 {currency_pair} 的数据")
            
            df = self.pairs_data[currency_pair].copy()
            
            # 日期过滤
            if start_date and end_date:
                df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
            
            if df.empty:
                raise Exception("选定日期范围内没有数据")
            
            # 确保数据点足够密集，如果数据点太少，可以进行插值
            if len(df) < 100:  # 如果数据点少于100个
                # 创建完整的日期范围
                date_range = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='D')
                # 创建新的DataFrame
                new_df = pd.DataFrame({'Date': date_range})
                # 合并原始数据
                df = pd.merge(new_df, df, on='Date', how='left')
                # 填充缺失值
                df = df.fillna(method='ffill').fillna(method='bfill')
            
            # 初始化变量
            equity = [self.initial_capital]  # 从初始资金开始
            position = 0
            
            # 执行回测
            for i in range(1, len(df)):
                if df['Signal'].iloc[i-1] > 0:
                    position = 1
                elif df['Signal'].iloc[i-1] < 0:
                    position = -1
                else:
                    position = 0
                
                daily_return = position * (df['Price'].iloc[i] / df['Price'].iloc[i-1] - 1)
                equity.append(equity[-1] * (1 + daily_return))
            
            # 计算回测指标
            equity_curve = pd.Series(equity, index=df.index)
            
            # 确保返回足够多的数据点
            equity_data = {}
            for i, (date, value) in enumerate(zip(df['Date'], equity)):
                # 每天都返回数据点
                equity_data[str(date.date())] = float(value)
            
            returns = pd.Series([(eq - equity[i-1])/equity[i-1] for i, eq in enumerate(equity[1:], 1)], index=df.index[1:])
            
            total_return = (equity[-1] / self.initial_capital - 1)
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 0 else 0
            max_drawdown = abs((equity_curve / equity_curve.cummax() - 1).min())
            win_rate = len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0
            
            return {
                'success': True,
                'total_return': float(total_return),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown),
                'win_rate': float(win_rate),
                'equity_curve': equity_data
            }
            
        except Exception as e:
            self.logger.error(f"回测执行错误: {str(e)}")
            return {'success': False, 'error': str(e)}

    def _calculate_indicators(self, df, params):
        """计算各种技术指标"""
        # 移动平均线
        for period in [5, 10, 20, 30, 50, 100, 200]:
            df[f'SMA{period}'] = df['Price'].rolling(period).mean()
        
        # 指数移动平均线
        for period in [12, 26, 50, 200]:
            df[f'EMA{period}'] = df['Price'].ewm(span=period, adjust=False).mean()
        
        # 波动率
        df['Volatility'] = df['Price'].pct_change().rolling(20).std() * np.sqrt(252)
        
        # RSI
        delta = df['Price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, 0.001)  # 避免除以零
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['EMA12'] = df['Price'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Price'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
        
        # 布林带
        for period in [20, 50]:
            df[f'BB_Middle_{period}'] = df['Price'].rolling(period).mean()
            std_dev = df['Price'].rolling(period).std()
            df[f'BB_Upper_{period}'] = df[f'BB_Middle_{period}'] + (std_dev * 2)
            df[f'BB_Lower_{period}'] = df[f'BB_Middle_{period}'] - (std_dev * 2)
            df[f'BB_Width_{period}'] = (df[f'BB_Upper_{period}'] - df[f'BB_Lower_{period}']) / df[f'BB_Middle_{period}']
            df[f'BB_%B_{period}'] = (df['Price'] - df[f'BB_Lower_{period}']) / (df[f'BB_Upper_{period}'] - df[f'BB_Lower_{period}'])
        
        # ADX (平均方向指数)
        if params.get('use_adx_filter', False):
            high = df['Price'].rolling(2).max()
            low = df['Price'].rolling(2).min()
            
            # 计算+DM和-DM
            plus_dm = high.diff()
            minus_dm = low.diff(-1).abs()
            
            # 修正+DM和-DM
            plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
            minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)
            
            # 计算TR (真实波幅)
            tr1 = high - low
            tr2 = (high - df['Price'].shift(1)).abs()
            tr3 = (low - df['Price'].shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # 计算平滑的+DM、-DM和TR
            smoothing_period = 14
            smooth_plus_dm = plus_dm.rolling(smoothing_period).sum()
            smooth_minus_dm = minus_dm.rolling(smoothing_period).sum()
            smooth_tr = tr.rolling(smoothing_period).sum()
            
            # 计算+DI和-DI
            plus_di = 100 * (smooth_plus_dm / smooth_tr)
            minus_di = 100 * (smooth_minus_dm / smooth_tr)
            
            # 计算DX和ADX
            dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
            df['ADX'] = dx.rolling(smoothing_period).mean()
            df['Plus_DI'] = plus_di
            df['Minus_DI'] = minus_di

    def _apply_mean_reversion_strategy(self, df, i, original_signal, params):
        """应用均值回归策略 - 专为CNYAUD设计"""
        # 获取当前价格
        price = df['Price'].iloc[i]
        
        # 使用布林带
        if params['use_bollinger']:
            period = params['lookback_period']
            bb_middle = df[f'BB_Middle_{period}'].iloc[i]
            bb_upper = df[f'BB_Upper_{period}'].iloc[i]
            bb_lower = df[f'BB_Lower_{period}'].iloc[i]
            
            # 计算价格相对于布林带的位置
            percent_b = df[f'BB_%B_{period}'].iloc[i]
            
            # 均值回归信号
            if percent_b > 1.0:  # 价格高于上轨
                signal = -1  # 卖出信号
            elif percent_b < 0.0:  # 价格低于下轨
                signal = 1  # 买入信号
            else:
                signal = 0  # 无信号
            
            # 使用RSI过滤
            if params['use_rsi_filter']:
                rsi = df['RSI'].iloc[i]
                
                # RSI超买/超卖确认
                if signal == 1 and rsi > params['rsi_oversold']:  # 买入信号但RSI不是超卖
                    signal = 0
                elif signal == -1 and rsi < params['rsi_overbought']:  # 卖出信号但RSI不是超买
                    signal = 0
            
            # 信号反转 - 如果启用
            if params['use_signal_reversal'] and original_signal != 0:
                # 如果原始信号与均值回归信号相反，增强信号
                if (original_signal > 0 and signal < 0) or (original_signal < 0 and signal > 0):
                    signal = signal * 1.5  # 增强反向信号
        else:
            # 如果不使用布林带，使用简单的均值回归
            sma = df[f'SMA{params["lookback_period"]}'].iloc[i]
            std_dev = df['Price'].iloc[i-params['lookback_period']:i].std()
            
            # 如果价格偏离均线超过标准差阈值，生成反向信号
            if price > sma + (std_dev * params['std_dev_threshold']):
                signal = -1  # 卖出信号
            elif price < sma - (std_dev * params['std_dev_threshold']):
                signal = 1  # 买入信号
            else:
                signal = 0  # 无信号
        
        return signal

    def _apply_trend_following_strategy(self, df, i, original_signal, params):
        """应用趋势跟踪策略 - 专为CNYUSD设计"""
        # 使用均线交叉
        if params['use_ma_crossover']:
            fast_ma = df[f'SMA{params["fast_ma"]}'].iloc[i]
            slow_ma = df[f'SMA{params["slow_ma"]}'].iloc[i]
            
            # 前一天的均线
            prev_fast_ma = df[f'SMA{params["fast_ma"]}'].iloc[i-1]
            prev_slow_ma = df[f'SMA{params["slow_ma"]}'].iloc[i-1]
            
            # 均线交叉信号
            if prev_fast_ma <= prev_slow_ma and fast_ma > slow_ma:
                signal = 1  # 金叉买入
            elif prev_fast_ma >= prev_slow_ma and fast_ma < slow_ma:
                signal = -1  # 死叉卖出
            else:
                signal = original_signal  # 保持原始信号
            
            # 使用ADX过滤
            if params['use_adx_filter']:
                adx = df['ADX'].iloc[i]
                plus_di = df['Plus_DI'].iloc[i]
                minus_di = df['Minus_DI'].iloc[i]
                
                # 只在强趋势中交易
                if adx < params['adx_threshold']:
                    signal = 0  # ADX低于阈值，无趋势
                elif signal > 0 and plus_di < minus_di:
                    signal = 0  # 买入信号但+DI小于-DI
                elif signal < 0 and plus_di > minus_di:
                    signal = 0  # 卖出信号但+DI大于-DI
            
            # 使用MACD确认
            if params['use_macd']:
                macd = df['MACD'].iloc[i]
                signal_line = df['Signal_Line'].iloc[i]
                macd_hist = df['MACD_Hist'].iloc[i]
                
                # MACD确认
                if signal > 0 and (macd < signal_line or macd_hist < 0):
                    signal = 0  # 买入信号但MACD不支持
                elif signal < 0 and (macd > signal_line or macd_hist > 0):
                    signal = 0  # 卖出信号但MACD不支持
        else:
            # 如果不使用均线交叉，使用原始信号
            signal = original_signal
        
        # 波动率过滤
        if params['use_volatility_filter']:
            volatility = df['Volatility'].iloc[i]
            avg_volatility = df['Volatility'].iloc[i-params['volatility_lookback']:i].mean()
            
            # 只在波动率适中时交易
            if volatility > avg_volatility * 2:
                signal = 0  # 波动率过高，不交易
        
        return signal

    def _calculate_metrics(self, trades_df, equity_df):
        """计算回测指标"""
        try:
            # 计算收益率
            returns = equity_df['capital'].pct_change().fillna(0)
            
            # 计算总收益率
            total_return = (equity_df['capital'].iloc[-1] / equity_df['capital'].iloc[0]) - 1
            
            # 计算年化收益率
            days = (equity_df.index[-1] - equity_df.index[0]).days
            annual_return = (1 + total_return) ** (365 / max(days, 1)) - 1 if days > 0 else 0
            
            # 计算最大回撤
            max_drawdown = equity_df['drawdown'].max()
            
            # 计算夏普比率
            risk_free_rate = 0.02  # 假设无风险利率为2%
            daily_risk_free = risk_free_rate / 252
            excess_returns = returns - daily_risk_free
            sharpe_ratio = excess_returns.mean() / max(excess_returns.std(), 1e-10) * np.sqrt(252)
            
            # 计算索提诺比率
            downside_returns = returns[returns < 0]
            sortino_ratio = returns.mean() / max(downside_returns.std(), 1e-10) * np.sqrt(252) if len(downside_returns) > 0 else 0
            
            # 计算卡玛比率
            calmar_ratio = annual_return / max(max_drawdown, 1e-10) if max_drawdown > 0 else 0
            
            # 交易统计
            if not trades_df.empty and 'returns' in trades_df.columns:
                # 排除开仓交易（收益为0）
                closed_trades = trades_df[trades_df['returns'] != 0]
                
                # 计算胜率
                winning_trades = closed_trades[closed_trades['returns'] > 0]
                win_rate = len(winning_trades) / max(len(closed_trades), 1) if len(closed_trades) > 0 else 0
                
                # 计算平均收益
                avg_trade = closed_trades['returns'].mean() if len(closed_trades) > 0 else 0
                
                # 计算平均盈利和亏损
                avg_win = winning_trades['returns'].mean() if len(winning_trades) > 0 else 0
                losing_trades = closed_trades[closed_trades['returns'] < 0]
                avg_loss = losing_trades['returns'].mean() if len(losing_trades) > 0 else 0
                
                # 计算盈亏比
                win_sum = winning_trades['returns'].sum() if len(winning_trades) > 0 else 0
                loss_sum = abs(losing_trades['returns'].sum()) if len(losing_trades) > 0 else 1e-10
                profit_factor = win_sum / loss_sum if loss_sum > 0 else win_sum if win_sum > 0 else 0
                
                # 计算最大连续亏损
                max_consecutive_losses = 0
                current_streak = 0
                for _, trade in closed_trades.iterrows():
                    if trade['returns'] < 0:
                        current_streak += 1
                        max_consecutive_losses = max(max_consecutive_losses, current_streak)
                    else:
                        current_streak = 0
            else:
                win_rate = 0
                avg_trade = 0
                avg_win = 0
                avg_loss = 0
                profit_factor = 0
                max_consecutive_losses = 0
            
            return {
                'equity_curve': equity_df,
                'trades': trades_df,
                'total_return': total_return,
                'annual_return': annual_return,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'win_rate': win_rate,
                'avg_trade': avg_trade,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'max_consecutive_losses': max_consecutive_losses
            }
        except Exception as e:
            self.logger.error(f"计算指标时出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # 返回默认指标
            return self._get_default_metrics()

    def _get_default_metrics(self):
        """返回默认的指标值"""
        return {
            'trades': pd.DataFrame(),
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
        """为单个货币对生成分析图表"""
        try:
            # 设置更美观的字体和样式
            plt.style.use('seaborn-v0_8-whitegrid')
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
            plt.rcParams['font.size'] = 10
            plt.rcParams['axes.titlesize'] = 14
            plt.rcParams['axes.labelsize'] = 12
            
            # 创建一个大图表
            fig = plt.figure(figsize=(16, 12), dpi=100)
            
            # 设置网格布局
            gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1])
            
            # 1. 权益曲线
            ax1 = plt.subplot(gs[0, :])
            equity_curve = result['equity_curve']['capital']
            equity_curve.plot(ax=ax1, color='#1f77b4', linewidth=2)
            ax1.set_title(f'{pair} 权益曲线', fontweight='bold', color='#333333')
            ax1.set_ylabel('资金', color='#555555')
            ax1.grid(True, alpha=0.3)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            
            # 添加起始和结束资金标注
            start_capital = equity_curve.iloc[0]
            end_capital = equity_curve.iloc[-1]
            ax1.annotate(f'起始: {start_capital:.2f}', 
                        xy=(equity_curve.index[0], start_capital),
                        xytext=(10, 10), textcoords='offset points',
                        color='#555555', fontsize=9)
            ax1.annotate(f'结束: {end_capital:.2f}', 
                        xy=(equity_curve.index[-1], end_capital),
                        xytext=(-60, 10), textcoords='offset points',
                        color='green' if end_capital > start_capital else 'red', 
                        fontsize=9, fontweight='bold')
            
            # 2. 回撤图
            ax2 = plt.subplot(gs[1, 0])
            drawdown = result['equity_curve']['drawdown']
            drawdown.plot(ax=ax2, color='#d62728', linewidth=1.5)
            ax2.fill_between(drawdown.index, 0, drawdown, color='#d62728', alpha=0.3)
            ax2.set_title('回撤分析', fontweight='bold', color='#333333')
            ax2.set_ylabel('回撤比例', color='#555555')
            ax2.grid(True, alpha=0.3)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            
            # 标记最大回撤
            max_dd = drawdown.max()
            max_dd_idx = drawdown.idxmax()
            ax2.annotate(f'最大回撤: {max_dd*100:.2f}%', 
                        xy=(max_dd_idx, max_dd),
                        xytext=(10, 10), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color='#555555'),
                        color='#d62728', fontsize=9, fontweight='bold')
            
            # 3. 交易分布
            ax3 = plt.subplot(gs[1, 1])
            
            if 'trades' in result and not result['trades'].empty and 'returns' in result['trades'].columns:
                # 筛选出平仓交易
                closed_trades = result['trades'][result['trades']['returns'] != 0]
                
                if not closed_trades.empty:
                    # 绘制交易收益分布
                    returns = closed_trades['returns']
                    ax3.hist(returns, bins=20, color='#2ca02c', alpha=0.7)
                    ax3.axvline(0, color='#d62728', linestyle='--', alpha=0.7)
                    ax3.set_title('交易收益分布', fontweight='bold', color='#333333')
                    ax3.set_xlabel('收益率', color='#555555')
                    ax3.set_ylabel('交易次数', color='#555555')
                    ax3.spines['top'].set_visible(False)
                    ax3.spines['right'].set_visible(False)
                    
                    # 添加平均收益标注
                    avg_return = returns.mean()
                    ax3.axvline(avg_return, color='#ff7f0e', linestyle='-', alpha=0.7)
                    ax3.annotate(f'平均收益: {avg_return*100:.2f}%', 
                                xy=(avg_return, ax3.get_ylim()[1]*0.9),
                                xytext=(10, -20) if avg_return > 0 else (-60, -20), 
                                textcoords='offset points',
                                color='#ff7f0e', fontsize=9, fontweight='bold')
                else:
                    ax3.text(0.5, 0.5, '没有平仓交易', ha='center', va='center', color='#555555')
            else:
                ax3.text(0.5, 0.5, '没有交易数据', ha='center', va='center', color='#555555')
            
            # 4. 月度收益热图 - 修复版本
            ax4 = plt.subplot(gs[2, 0])
            
            try:
                if 'equity_curve' in result and len(result['equity_curve']) > 30:
                    equity_curve = result['equity_curve']['capital']
                    
                    # 计算月度收益率
                    monthly_returns = equity_curve.resample('M').last().pct_change().dropna()
                    
                    if len(monthly_returns) > 0:
                        # 创建年月索引的DataFrame
                        years = sorted(monthly_returns.index.year.unique())
                        months = range(1, 13)
                        
                        # 创建空的月度收益表
                        heatmap_data = np.zeros((len(years), 12))
                        heatmap_data[:] = np.nan
                        
                        # 填充数据
                        for i, year in enumerate(years):
                            for j, month in enumerate(months):
                                try:
                                    # 查找对应年月的数据
                                    date_str = f"{year}-{month:02d}"
                                    matches = [date for date in monthly_returns.index if date.strftime('%Y-%m').startswith(date_str)]
                                    if matches:
                                        heatmap_data[i, j] = monthly_returns.loc[matches[0]]
                                except Exception as e:
                                    self.logger.warning(f"处理月度数据时出错: {str(e)}")
                        
                        # 创建DataFrame
                        heatmap_df = pd.DataFrame(heatmap_data, index=years, columns=range(1, 13))
                        
                        # 绘制热图
                        sns.heatmap(heatmap_df, ax=ax4, cmap='RdYlGn', center=0, 
                                   annot=True, fmt='.1%', linewidths=.5, cbar=False,
                                   annot_kws={"size": 8})
                        
                        ax4.set_title('月度收益热图', fontweight='bold', color='#333333')
                        ax4.set_ylabel('年份', color='#555555')
                        ax4.set_xlabel('月份', color='#555555')
                        
                        # 设置月份标签
                        month_labels = ['一月', '二月', '三月', '四月', '五月', '六月', 
                                       '七月', '八月', '九月', '十月', '十一月', '十二月']
                        ax4.set_xticklabels(month_labels, rotation=45, ha='right')
                    else:
                        # 替代方案：绘制季度收益柱状图
                        self._draw_quarterly_returns(ax4, result['equity_curve']['capital'])
                else:
                    # 替代方案：绘制季度收益柱状图
                    self._draw_quarterly_returns(ax4, result['equity_curve']['capital'])
                    
            except Exception as e:
                self.logger.error(f"生成月度热图时出错: {str(e)}")
                # 替代方案：绘制季度收益柱状图
                self._draw_quarterly_returns(ax4, result['equity_curve']['capital'])
            
            # 5. 回测指标汇总
            ax5 = plt.subplot(gs[2, 1])
            ax5.axis('off')
            
            # 创建一个带有圆角和阴影的背景
            rect = plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=True, alpha=0.1, 
                                color='#f0f0f0', transform=ax5.transAxes, 
                                zorder=-1, ec='#cccccc', lw=1)
            ax5.add_patch(rect)
            
            # 将指标分组
            metrics = {
                '收益指标': [
                    ('总收益率', f"{result['total_return']*100:.2f}%"),
                    ('年化收益率', f"{result['annual_return']*100:.2f}%"),
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
            ax5.text(0.5, 0.95, "回测指标汇总", fontsize=14, fontweight='bold', 
                     ha='center', va='top', color='#333333')
            
            # 绘制指标组
            y_pos = 0.85
            for group_name, group_metrics in metrics.items():
                # 绘制组标题
                ax5.text(0.1, y_pos, group_name, fontsize=12, fontweight='bold',
                        color='#1f77b4')
                
                # 添加分隔线
                ax5.axhline(y=y_pos-0.02, xmin=0.1, xmax=0.9, color='#1f77b4', alpha=0.3, lw=1)
                
                y_pos -= 0.06
                
                # 绘制组内指标
                for name, value in group_metrics:
                    # 使用不同的颜色标记正负值
                    if '%' in value and value != '0.00%':
                        try:
                            val_num = float(value.strip('%'))
                            if val_num > 0:
                                value_color = '#2ca02c'  # 绿色
                                value = f"▲ {value}"
                            elif val_num < 0:
                                value_color = '#d62728'  # 红色
                                value = f"▼ {value}"
                            else:
                                value_color = '#555555'  # 灰色
                        except:
                            value_color = '#555555'
                    else:
                        value_color = '#555555'
                    
                    ax5.text(0.15, y_pos, f"{name}:", fontsize=10, color='#333333')
                    ax5.text(0.5, y_pos, value, fontsize=10, color=value_color, fontweight='bold')
                    y_pos -= 0.05
                
                # 组间距
                y_pos -= 0.02
            
            # 添加时间范围信息
            start_date = result['equity_curve'].index[0].strftime('%Y-%m-%d')
            end_date = result['equity_curve'].index[-1].strftime('%Y-%m-%d')
            ax5.text(0.1, 0.05, f"回测区间: {start_date} 至 {end_date}", 
                     fontsize=9, color='#777777', style='italic')
            
            # 添加交易次数信息
            trade_count = len(result['trades']) if 'trades' in result else 0
            ax5.text(0.7, 0.05, f"交易次数: {trade_count}", 
                     fontsize=9, color='#777777', style='italic')
            
            # 调整布局并保存
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{pair}_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"已生成 {pair} 的分析图表")
            
        except Exception as e:
            self.logger.error(f"生成{pair}分析图表失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _draw_quarterly_returns(self, ax, equity_curve):
        """绘制季度收益柱状图作为月度热图的替代"""
        try:
            # 计算季度收益
            quarterly_returns = equity_curve.resample('Q').last().pct_change().dropna()
            
            if len(quarterly_returns) > 0:
                # 创建季度标签
                quarters = [f"{date.year}Q{(date.month-1)//3+1}" for date in quarterly_returns.index]
                
                # 绘制柱状图
                bars = ax.bar(quarters, quarterly_returns.values * 100, 
                             color=[('#2ca02c' if x > 0 else '#d62728') for x in quarterly_returns.values])
                
                # 添加数值标签
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., 
                           height + (1 if height > 0 else -3),
                           f'{height:.1f}%',
                           ha='center', va='bottom' if height > 0 else 'top',
                           color='#333333', fontsize=8)
                
                ax.set_title('季度收益表现', fontweight='bold', color='#333333')
                ax.set_ylabel('收益率 (%)', color='#555555')
                ax.set_xlabel('季度', color='#555555')
                
                # 设置x轴标签
                if len(quarters) > 12:
                    # 如果季度太多，只显示部分标签
                    step = max(1, len(quarters) // 12)
                    ax.set_xticks(range(0, len(quarters), step))
                    ax.set_xticklabels([quarters[i] for i in range(0, len(quarters), step)], rotation=45, ha='right')
                else:
                    ax.set_xticklabels(quarters, rotation=45, ha='right')
                
                # 添加零线
                ax.axhline(y=0, color='#888888', linestyle='-', alpha=0.3)
                
                # 美化
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(axis='y', alpha=0.3)
            else:
                ax.text(0.5, 0.5, '数据不足，无法生成季度收益图', 
                       ha='center', va='center', color='#555555')
        except Exception as e:
            self.logger.error(f"绘制季度收益图时出错: {str(e)}")
            ax.text(0.5, 0.5, '生成季度收益图失败', 
                   ha='center', va='center', color='#555555')

def main():
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # 创建回测器实例
        backtester = ForexBacktester()
        
        # 加载数据
        if not backtester.load_data():
            logger.error("数据加载失败")
            return
        
        # 执行回测（添加必要的参数）
        results = backtester.run_backtest(
            currency_pair="CNYUSD",  # 指定货币对
            start_date="2015-01-28", # 指定开始日期
            end_date="2024-12-31"    # 指定结束日期
        )
        
        if results['success']:
            logger.info("回测完成，结果：")
            logger.info(f"总收益率: {results['total_return']:.2%}")
            logger.info(f"夏普比率: {results['sharpe_ratio']:.2f}")
            logger.info(f"最大回撤: {results['max_drawdown']:.2%}")
            logger.info(f"胜率: {results['win_rate']:.2%}")
        
    except Exception as e:
        logger.error(f"回测过程发生错误: {str(e)}")

if __name__ == "__main__":
    main()
