def calculate_portfolio_risk(self, df_dict):
    """计算投资组合风险指标"""
    try:
        returns = pd.DataFrame()

        # 计算各货币对收益率
        for pair, df in df_dict.items():
            returns[pair] = df['Close'].pct_change()

        # 计算相关性矩阵
        corr_matrix = returns.corr()

        # 计算波动率
        volatility = returns.std() * np.sqrt(252)  # 年化波动率

        # 计算VaR (95% 置信度)
        var_95 = returns.quantile(0.05)

        # 计算CVaR
        cvar_95 = returns[returns <= var_95].mean()

        # 计算夏普比率 (假设无风险利率为2%)
        rf = 0.02
        excess_returns = returns.mean() * 252 - rf
        sharpe_ratio = excess_returns / volatility

        # 风险分布
        risk_dist = {
            'low_risk': (returns > -0.01).mean(),
            'medium_risk': ((returns <= -0.01) & (returns > -0.02)).mean(),
            'high_risk': (returns <= -0.02).mean()
        }

        return {
            'correlation_matrix': corr_matrix.round(4).to_dict(),
            'volatility': volatility.round(4).to_dict(),
            'var_95': var_95.round(4).to_dict(),
            'cvar_95': cvar_95.round(4).to_dict(),
            'sharpe_ratio': sharpe_ratio.round(4).to_dict(),
            'risk_distribution': {k: v.round(4).to_dict() for k, v in risk_dist.items()}
        }

    except Exception as e:
        self.logger.error(f"计算投资组合风险指标时出错: {str(e)}")
        return None

def run_optimized_backtest(self, df, currency_pair, params=None):

    """运行优化后的回测"""
    try:
        # 运行基准回测（使用默认参数）
        default_params = {
            'stop_loss': 0.02,
            'take_profit': 0.03,
            'position_size': 1000,
            'initial_capital': 100000,
            'commission': 0.0002,
            'slippage': 0.0001
        }

        baseline_results = self.run_backtest(df, currency_pair, default_params)
        if not baseline_results['success']:
            return baseline_results

        # 运行优化后的回测
        optimized_results = self.run_backtest(df, currency_pair, params)
        if not optimized_results['success']:
            return optimized_results

        # 计算累计收益率
        baseline_trades = baseline_results['data']['trades']
        optimized_trades = optimized_results['data']['trades']

        # 获取所有交易日期
        all_dates = sorted(list(set(
            [trade['date'] for trade in baseline_trades] +
            [trade['date'] for trade in optimized_trades]
        )))

        # 计算累计收益
        baseline_returns = self._calculate_cumulative_returns(baseline_trades, all_dates)
        optimized_returns = self._calculate_cumulative_returns(optimized_trades, all_dates)

        # 合并结果
        return {
            'success': True,
            'data': {
                'trades': optimized_trades,
                'metrics': optimized_results['data']['metrics'],
                'baseline_results': baseline_results['data']['metrics'],
                'optimization_results': {
                    'best_params': params,
                    'best_metrics': optimized_results['data']['metrics']
                },
                'cumulative_returns': {
                    'dates': all_dates,
                    'baseline': baseline_returns,
                    'optimized': optimized_returns
                }
            }
        }

    except Exception as e:
        self.logger.error(f"运行优化回测时出错: {str(e)}")
        return {
            'success': False,
            'error': f"运行优化回测时出错: {str(e)}"
        }


def _calculate_cumulative_returns(self, trades, dates):
    """计算累计收益率"""
    try:
        # 初始化累计收益率字典
        cumulative_returns = {}
        current_return = 0

        # 计算每个交易的收益率并累加
        for trade in trades:
            trade_date = trade['date']
            trade_return = trade['pnl'] / trade['balance'] if trade['balance'] != 0 else 0
            current_return += trade_return
            cumulative_returns[trade_date] = current_return * 100  # 转换为百分比

        # 对所有日期进行填充
        returns_series = []
        last_return = 0
        for date in dates:
            if date in cumulative_returns:
                last_return = cumulative_returns[date]
            returns_series.append(last_return)

        return returns_series

    except Exception as e:
        self.logger.error(f"计算累计收益率时出错: {str(e)}")
        return [0] * len(dates)