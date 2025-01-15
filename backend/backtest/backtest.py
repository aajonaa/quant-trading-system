import pandas as pd
import numpy as np
import logging
from datetime import datetime

class Backtester:
    """回测器类"""
    
    def __init__(self):
        self.trades = []
        self.current_position = None
        self.balance = 0
        self.initial_capital = 0
        np.random.seed(42)  # 设置固定的随机种子
        
    def run_backtest(self, df, signals, params):
        """
        运行回测
        
        Args:
            df: 历史数据DataFrame
            signals: 交易信号
            params: 回测参数
                - stop_loss: 止损比例
                - take_profit: 止盈比例
                - position_size: 仓位大小
                - initial_capital: 初始资金
                - commission: 手续费率
                - slippage: 滑点率
                - max_holding_days: 最大持仓天数
                - min_holding_days: 最小持仓天数
                
        Returns:
            dict: 回测结果
        """
        try:
            # 重置状态
            self.trades = []
            self.current_position = None
            self.balance = params.get('initial_capital', 100000)
            self.initial_capital = self.balance
            
            # 获取参数
            stop_loss = params.get('stop_loss', 0.02)
            take_profit = params.get('take_profit', 0.03)
            position_size = params.get('position_size', 1000)
            commission = params.get('commission', 0.0002)
            slippage = params.get('slippage', 0.0001)
            max_holding_days = params.get('max_holding_days', 15)  # 最大持仓天数
            min_holding_days = params.get('min_holding_days', 3)   # 最小持仓天数
            
            # 初始化变量
            balance = self.initial_capital
            position = 0
            entry_price = 0
            entry_date = None
            trades = []
            current_drawdown = 0
            max_drawdown = 0
            peak_balance = self.initial_capital
            
            # 遍历每个交易日
            for i in range(len(df)):
                current_date = df.index[i]
                current_price = df['Close'].iloc[i]
                
                # 更新最大回撤
                if balance > peak_balance:
                    peak_balance = balance
                current_drawdown = (peak_balance - balance) / peak_balance * 100
                max_drawdown = max(max_drawdown, current_drawdown)
                
                # 如果有持仓
                if position != 0:
                    # 计算持仓天数
                    holding_days = (current_date - entry_date).days
                    
                    # 计算收益率
                    if position > 0:
                        profit_pct = (current_price - entry_price) / entry_price
                    else:
                        profit_pct = (entry_price - current_price) / entry_price
                    
                    # 检查是否需要平仓
                    should_close = False
                    exit_reason = None
                    
                    # 止损检查
                    if profit_pct <= -stop_loss:
                        should_close = True
                        exit_reason = 'stop_loss'
                    
                    # 止盈检查
                    elif profit_pct >= take_profit:
                        should_close = True
                        exit_reason = 'take_profit'
                    
                    # 最大持仓时间检查
                    elif holding_days >= max_holding_days:
                        should_close = True
                        exit_reason = 'time_limit'
                    
                    # 信号反转检查（需要满足最小持仓时间）
                    elif holding_days >= min_holding_days:
                        if (position > 0 and signals[i] < 0) or (position < 0 and signals[i] > 0):
                            should_close = True
                            exit_reason = 'signal_reverse'
                    
                    # 执行平仓
                    if should_close:
                        # 计算交易成本
                        trading_cost = abs(position * current_price * (commission + slippage))
                        
                        # 更新账户余额
                        if position > 0:
                            profit = position * (current_price - entry_price) - trading_cost
                        else:
                            profit = -position * (entry_price - current_price) - trading_cost
                        
                        balance += profit
                        
                        # 记录交易
                        trades.append({
                            'date': current_date,
                            'type': 'long' if position > 0 else 'short',
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'profit': profit,
                            'profit_pct': profit_pct * 100,
                            'balance': balance,
                            'holding_days': holding_days,
                            'exit_reason': exit_reason
                        })
                        
                        # 清空持仓
                        position = 0
                        entry_price = 0
                        entry_date = None
                
                # 如果没有持仓且有交易信号
                elif signals[i] != 0:
                    # 计算交易成本
                    trading_cost = position_size * current_price * (commission + slippage)
                    
                    # 开仓
                    if signals[i] > 0:  # 做多信号
                        position = position_size
                        entry_price = current_price
                        balance -= trading_cost
                    elif signals[i] < 0:  # 做空信号
                        position = -position_size
                        entry_price = current_price
                        balance -= trading_cost
                    
                    entry_date = current_date
            
            # 计算回测指标
            if len(trades) > 0:
                # 计算总收益率
                total_return = (balance - self.initial_capital) / self.initial_capital * 100
                
                # 计算胜率
                profitable_trades = sum(1 for trade in trades if trade['profit'] > 0)
                win_rate = profitable_trades / len(trades) * 100
                
                # 计算平均收益
                avg_profit = sum(trade['profit'] for trade in trades if trade['profit'] > 0) / profitable_trades if profitable_trades > 0 else 0
                avg_loss = sum(trade['profit'] for trade in trades if trade['profit'] <= 0) / (len(trades) - profitable_trades) if len(trades) - profitable_trades > 0 else 0
                
                # 计算平均持仓时间
                avg_trade_duration = sum(trade['holding_days'] for trade in trades) / len(trades)
                
                # 计算夏普比率
                returns = [trade['profit_pct'] for trade in trades]
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) != 0 else 0
                
                metrics = {
                    'total_return': total_return,
                    'max_drawdown': max_drawdown,
                    'win_rate': win_rate,
                    'total_trades': len(trades),
                    'profitable_trades': profitable_trades,
                    'avg_profit': avg_profit,
                    'avg_loss': avg_loss,
                    'profit_factor': abs(avg_profit / avg_loss) if avg_loss != 0 else float('inf'),
                    'sharpe_ratio': sharpe_ratio,
                    'avg_trade_duration': avg_trade_duration
                }
            else:
                metrics = {
                    'total_return': 0,
                    'max_drawdown': 0,
                    'win_rate': 0,
                    'total_trades': 0,
                    'profitable_trades': 0,
                    'avg_profit': 0,
                    'avg_loss': 0,
                    'profit_factor': 0,
                    'sharpe_ratio': 0,
                    'avg_trade_duration': 0
                }
            
            return {
                'success': True,
                'data': {
                    'metrics': metrics,
                    'trades': trades
                }
            }
            
        except Exception as e:
            self.logger.error(f"回测执行出错: {str(e)}")
            return {
                'success': False,
                'error': f"回测执行出错: {str(e)}"
            }
    
    def _open_position(self, trade_type, date, price, position_size, reason):
        """开仓"""
        self.current_position = {
            'type': trade_type,
            'entry_date': date,
            'entry_price': price,
            'position': position_size,
            'entry_reason': reason
        }
    
    def _close_position(self, date, price, reason):
        """平仓"""
        if self.current_position is None:
            return
            
        # 计算收益
        if self.current_position['type'] == 'buy':
            pnl = (price - self.current_position['entry_price']) * self.current_position['position']
        else:
            pnl = (self.current_position['entry_price'] - price) * self.current_position['position']
            
        # 更新余额
        self.balance += pnl
        
        # 记录交易
        trade = {
            'date': date,
            'type': self.current_position['type'],
            'price': price,
            'position': self.current_position['position'],
            'balance': self.balance,
            'pnl': pnl,
            'entry_reason': self.current_position['entry_reason'],
            'exit_reason': reason
        }
        self.trades.append(trade)
        
        # 清空当前仓位
        self.current_position = None