import numpy as np
import pandas as pd
import logging

class ForexBacktester:
    """外汇回测系统"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def run_backtest(self, df, params=None):
        """运行回测
        
        Args:
            df (pd.DataFrame): 包含 OHLCV 数据的 DataFrame
            params (dict): 回测参数字典
            
        Returns:
            dict: 回测结果
        """
        try:
            if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                self.logger.error("输入数据无效")
                return None
                
            # 确保必要的列存在
            required_columns = ['Open', 'High', 'Low', 'Close',  'Signal']
            if not all(col in df.columns for col in required_columns):
                self.logger.error(f"数据缺少必要的列: {[col for col in required_columns if col not in df.columns]}")
                return None
                
            # 设置默认参数
            default_params = {
                'stop_loss': 0.02,        # 止损比例
                'take_profit': 0.03,      # 止盈比例
                'position_size': 1000,    # 每笔交易金额
            }
            
            # 更新参数
            if params:
                default_params.update(params)
            params = default_params
            
            # 初始化结果变量
            initial_capital = 100000
            current_capital = initial_capital
            position = 0
            entry_price = 0
            trades = []
            equity_curve = [initial_capital]
            
            # 遍历数据
            for i in range(1, len(df)):
                current_price = df['Close'].iloc[i]
                current_date = df.index[i]
                
                # 检查止损止盈
                if position != 0:
                    price_change = (current_price - entry_price) / entry_price
                    pnl = position * params['position_size'] * price_change
                    
                    # 止损检查
                    if (position > 0 and price_change < -params['stop_loss']) or \
                       (position < 0 and price_change > params['stop_loss']):
                        current_capital += pnl
                        trades.append({
                            'type': 'long' if position > 0 else 'short',
                            'entry_date': entry_date,
                            'entry_price': entry_price,
                            'exit_date': current_date,
                            'exit_price': current_price,
                            'pnl': pnl,
                            'exit_reason': 'stop_loss'
                        })
                        position = 0
                        
                    # 止盈检查
                    elif (position > 0 and price_change > params['take_profit']) or \
                         (position < 0 and price_change < -params['take_profit']):
                        current_capital += pnl
                        trades.append({
                            'type': 'long' if position > 0 else 'short',
                            'entry_date': entry_date,
                            'entry_price': entry_price,
                            'exit_date': current_date,
                            'exit_price': current_price,
                            'pnl': pnl,
                            'exit_reason': 'take_profit'
                        })
                        position = 0
                
                # 交易信号检查
                signal = df['Signal'].iloc[i]
                
                if position == 0:  # 没有持仓时
                    if signal > 0:  # 做多信号
                        position = 1
                        entry_price = current_price
                        entry_date = current_date
                    elif signal < 0:  # 做空信号
                        position = -1
                        entry_price = current_price
                        entry_date = current_date
                
                elif position > 0 and signal < 0:  # 持多仓遇到做空信号
                    price_change = (current_price - entry_price) / entry_price
                    pnl = position * params['position_size'] * price_change
                    current_capital += pnl
                    trades.append({
                        'type': 'long',
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'exit_date': current_date,
                        'exit_price': current_price,
                        'pnl': pnl,
                        'exit_reason': 'signal'
                    })
                    position = -1  # 反转做空
                    entry_price = current_price
                    entry_date = current_date
                    
                elif position < 0 and signal > 0:  # 持空仓遇到做多信号
                    price_change = (current_price - entry_price) / entry_price
                    pnl = position * params['position_size'] * price_change
                    current_capital += pnl
                    trades.append({
                        'type': 'short',
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'exit_date': current_date,
                        'exit_price': current_price,
                        'pnl': pnl,
                        'exit_reason': 'signal'
                    })
                    position = 1  # 反转做多
                    entry_price = current_price
                    entry_date = current_date
                
                # 更新权益曲线
                if position != 0:
                    price_change = (current_price - entry_price) / entry_price
                    current_equity = current_capital + position * params['position_size'] * price_change
                else:
                    current_equity = current_capital
                equity_curve.append(current_equity)
            
            # 计算回测指标
            equity_curve = np.array(equity_curve)
            returns = np.diff(equity_curve) / equity_curve[:-1]
            
            total_trades = len(trades)
            if total_trades == 0:
                return {
                    'error': '回测期间没有产生交易'
                }
                
            winning_trades = len([t for t in trades if t['pnl'] > 0])
            total_pnl = sum(t['pnl'] for t in trades)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # 计算最大回撤
            peak = np.maximum.accumulate(equity_curve)
            drawdown = (peak - equity_curve) / peak
            max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
            
            # 计算夏普比率 (假设无风险利率为2%)
            risk_free_rate = 0.02
            excess_returns = returns - risk_free_rate / 252  # 转换为日收益率
            sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(returns) if len(returns) > 0 else 0
            
            # 计算盈亏比
            profit_factor = abs(total_pnl / sum(t['pnl'] for t in trades if t['pnl'] < 0)) if sum(t['pnl'] for t in trades if t['pnl'] < 0) != 0 else float('inf')
            
            results = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': total_trades - winning_trades,
                'win_rate': round(win_rate * 100, 2),
                'profit_factor': round(profit_factor, 2),
                'total_return': round((current_capital - initial_capital) / initial_capital * 100, 2),
                'max_drawdown': round(max_drawdown * 100, 2),
                'sharpe_ratio': round(sharpe_ratio, 2),
                'equity_curve': equity_curve.tolist(),
                'trades': trades
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"回测过程中出错: {str(e)}")
            return None

