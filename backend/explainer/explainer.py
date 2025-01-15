from datetime import datetime

class SignalExplainer:
    """信号解释生成器类"""
    
    def __init__(self):
        self.signal_patterns = {
            'trend_reversal': '趋势反转',
            'ma_cross': '均线交叉',
            'rsi_signal': 'RSI超买卖',
            'volatility_breakout': '波动突破',
            'trend_follow': '趋势跟随',
            'momentum': '动量信号',
            'pattern': '形态信号'
        }
        
        self.risk_levels = {
            'low_risk': '低风险',
            'medium_risk': '中等风险',
            'high_risk': '高风险'
        }
    
    def explain_signals(self, data, signals, risk_analysis=None, backtest_results=None):
        """
        综合分析并解释交易信号、风险和回测结果
        """
        explanation = []
        
        # 添加标题
        explanation.append("📊 外汇交易分析报告")
        explanation.append("=" * 50)
        
        # 1. 市场趋势分析
        explanation.append("\n📈 市场趋势分析")
        explanation.append("-" * 30)
        trend_analysis = self._analyze_market_trend(data)
        explanation.append(trend_analysis)
        
        # 2. 交易信号分析
        explanation.append("\n🎯 交易信号分析")
        explanation.append("-" * 30)
        signal_analysis = self._analyze_signals(signals, data)
        explanation.append(signal_analysis)
        
        # 3. 技术指标分析
        explanation.append("\n📊 技术指标分析")
        explanation.append("-" * 30)
        technical_analysis = self._analyze_technical_indicators(data)
        explanation.append(technical_analysis)
        
        # 4. 风险分析
        if risk_analysis:
            explanation.append("\n⚠️ 风险分析")
            explanation.append("-" * 30)
            risk_assessment = self._analyze_risk(risk_analysis)
            explanation.append(risk_assessment)
        
        # 5. 回测结果分析
        if backtest_results:
            explanation.append("\n📉 回测分析")
            explanation.append("-" * 30)
            backtest_analysis = self._analyze_backtest(backtest_results)
            explanation.append(backtest_analysis)
        
        # 6. 综合建议
        explanation.append("\n💡 交易建议")
        explanation.append("-" * 30)
        recommendations = self._generate_recommendations(data, signals, risk_analysis, backtest_results)
        explanation.append(recommendations)
        
        # 添加结尾
        explanation.append("\n" + "=" * 50)
        explanation.append("分析报告生成时间：" + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        return "\n".join(explanation)
    
    def _analyze_market_trend(self, data):
        """分析市场整体趋势"""
        try:
            # 计算近期和远期趋势
            recent_trend = data['Close'].iloc[-5:].pct_change().mean()
            long_term_trend = data['Close'].iloc[-20:].pct_change().mean()
            volatility = data['Close'].iloc[-20:].pct_change().std()
            
            trend_text = []
            
            # 趋势强度判断
            if abs(recent_trend) > volatility:
                strength = "强"
            else:
                strength = "弱"
            
            # 趋势方向判断
            if recent_trend > 0:
                trend_text.append(f"• 近期趋势：{strength}势上涨 (+{recent_trend*100:.2f}%)\n")
            else:
                trend_text.append(f"• 近期趋势：{strength}势下跌 (-{abs(recent_trend)*100:.2f}%)\n")
            
            # 趋势对比
            if (recent_trend > 0) != (long_term_trend > 0):
                trend_text.append("• 趋势背离：短期与长期趋势出现背离，注意趋势转换风险\n")
            
            # 趋势持续性分析
            ma_10 = data['MA_10'].iloc[-1]
            ma_50 = data['MA_50'].iloc[-1]
            price = data['Close'].iloc[-1]
            
            trend_text.append("均线系统分析：\n")
            if price > ma_10 > ma_50:
                trend_text.append("• 多头排列：价格位于双均线之上，上升趋势确立\n")
            elif price < ma_10 < ma_50:
                trend_text.append("• 空头排列：价格位于双均线之下，下降趋势确立\n")
            else:
                trend_text.append("• 趋势混沌：价格在均线之间波动，趋势不明确\n")
            
            return "".join(trend_text)
        except Exception as e:
            return f"❌ 市场趋势分析出错：{str(e)}"
    
    def _analyze_signals(self, signals, data):
        """分析交易信号"""
        try:
            signal_analysis = []
            
            # 统计信号
            buy_signals = len([s for s in signals if s == 1])
            sell_signals = len([s for s in signals if s == -1])
            hold_signals = len([s for s in signals if s == 0])
            
            # 最新信号
            latest_signal = signals[-1] if len(signals) > 0 else 0
            signal_text = "🔵 买入" if latest_signal == 1 else "🔴 卖出" if latest_signal == -1 else "⚪ 持有"
            
            # 信号强度分析
            signal_strength = self._analyze_signal_strength(data)
            
            signal_analysis.extend([
                f"当前信号：{signal_text}\n",
                f"信号强度：{signal_strength}\n",
                "\n信号统计：\n",
                f"• 买入信号：{buy_signals} 次\n",
                f"• 卖出信号：{sell_signals} 次\n",
                f"• 持有信号：{hold_signals} 次\n"
            ])
            
            return "".join(signal_analysis)
        except Exception as e:
            return f"❌ 交易信号分析出错：{str(e)}"
    
    def _analyze_signal_strength(self, data):
        """分析信号强度"""
        try:
            # 计算技术指标的一致性
            rsi = data['RSI'].iloc[-1]
            ma_cross = (data['MA_10'].iloc[-1] / data['MA_50'].iloc[-1] - 1) * 100
            volatility = data['Historical_Volatility'].iloc[-1]
            
            strength_factors = []
            
            # RSI分析
            if rsi > 70:
                strength_factors.append("RSI超买")
            elif rsi < 30:
                strength_factors.append("RSI超卖")
                
            # 均线分析
            if abs(ma_cross) > 1:
                direction = "上穿" if ma_cross > 0 else "下穿"
                strength_factors.append(f"均线{direction}（{abs(ma_cross):.2f}%）")
                
            # 波动性分析
            if volatility > 20:
                strength_factors.append("高波动")
            elif volatility < 10:
                strength_factors.append("低波动")
            
            if not strength_factors:
                return "中性"
            return " | ".join(strength_factors)
            
        except Exception as e:
            return "❌ 信号强度分析出错"
    
    def _analyze_technical_indicators(self, data):
        """分析技术指标"""
        try:
            analysis = []
            
            # RSI分析
            rsi = data['RSI'].iloc[-1]
            rsi_status = "⚠️ 超买" if rsi > 70 else "💡 超卖" if rsi < 30 else "✅ 中性"
            analysis.append(f"• RSI指标：{rsi:.2f} [{rsi_status}]\n")
            
            # 均线分析
            ma_10 = data['MA_10'].iloc[-1]
            ma_50 = data['MA_50'].iloc[-1]
            price = data['Close'].iloc[-1]
            
            ma_cross = (ma_10 / ma_50 - 1) * 100
            if abs(ma_cross) > 0.1:
                cross_direction = "上穿" if ma_cross > 0 else "下穿"
                analysis.append(f"• 均线系统：MA10 {cross_direction} MA50（{abs(ma_cross):.2f}%）\n")
            
            # 波动性分析
            volatility = data['Historical_Volatility'].iloc[-1]
            atr = data['ATR'].iloc[-1]
            
            volatility_level = "⚠️ 高" if volatility > 20 else "💡 低" if volatility < 10 else "✅ 中等"
            analysis.append(f"\n波动性指标：\n")
            analysis.append(f"• 历史波动率：{volatility:.2f}% [{volatility_level}]\n")
            analysis.append(f"• ATR指标：{atr:.4f}\n")
            
            return "".join(analysis)
        except Exception as e:
            return f"❌ 技术指标分析出错：{str(e)}"
    
    def _analyze_risk(self, risk_analysis):
        """分析风险指标"""
        try:
            if not risk_analysis.get('success', False):
                return "❌ 风险分析数据无效"
                
            risk_data = risk_analysis.get('data', {})
            portfolio_risk = risk_data.get('portfolio_risk', {})
            risk_signals = risk_data.get('risk_signals', {})
            
            analysis = []
            
            # 组合风险分析
            if portfolio_risk:
                analysis.extend([
                    "📊 投资组合风险指标\n",
                    f"• 年化收益率：{portfolio_risk.get('annual_return', 0):.2f}%\n",
                    f"• 组合波动率：{portfolio_risk.get('portfolio_volatility', 0):.2f}%\n",
                    f"• 夏普比率：{portfolio_risk.get('sharpe_ratio', 0):.2f}\n",
                    f"• 95% VaR：{portfolio_risk.get('var_95', 0):.2f}%\n",
                    f"• 95% CVaR：{portfolio_risk.get('cvar_95', 0):.2f}%\n",
                    f"• 风险等级：{portfolio_risk.get('risk_level', '未知')}\n"
                ])
            
            # 各货币对风险分析
            if risk_signals:
                analysis.append("\n📈 各货币对风险状况\n")
                for pair, signal in risk_signals.items():
                    analysis.append(f"\n{pair}：\n")
                    analysis.append(f"• 风险评分：{signal.get('risk_score', 0):.2f}\n")
                    analysis.append(f"• 风险等级：{signal.get('risk_level', '未知')}\n")
                    analysis.append(f"• 信号：{signal.get('signal', '未知')}\n")
            
            return "".join(analysis)
        except Exception as e:
            return f"❌ 风险分析出错：{str(e)}"
    
    def _analyze_backtest(self, backtest_results):
        """分析回测结果"""
        try:
            if not backtest_results.get('success', False):
                return "❌ 回测结果无效"
                
            metrics = backtest_results.get('data', {}).get('metrics', {})
            trades = backtest_results.get('data', {}).get('trades', [])
            
            analysis = []
            
            # 回测指标分析
            if metrics:
                analysis.extend([
                    "📊 回测绩效指标\n",
                    f"• 总交易次数：{metrics.get('total_trades', 0)}\n",
                    f"• 盈利交易：{metrics.get('profitable_trades', 0)}\n",
                    f"• 胜率：{metrics.get('win_rate', 0):.2f}%\n",
                    f"• 总收益率：{metrics.get('total_return', 0):.2f}%\n",
                    f"• 最大回撤：{metrics.get('max_drawdown', 0):.2f}%\n",
                    f"• 夏普比率：{metrics.get('sharpe_ratio', 0):.2f}\n",
                    f"• 盈亏比：{metrics.get('profit_factor', 0):.2f}\n",
                    f"• 平均盈利：{metrics.get('average_profit', 0):.2f}%\n",
                    f"• 平均亏损：{metrics.get('average_loss', 0):.2f}%\n",
                    f"• 风险收益比：{metrics.get('risk_reward_ratio', 0):.2f}\n"
                ])
            
            # 最近交易分析
            if trades:
                recent_trades = trades[-3:]  # 最近3笔交易
                analysis.append("\n📝 最近交易记录\n")
                for trade in recent_trades:
                    trade_type = "🔵 买入" if trade.get('type') == 'buy' else "🔴 卖出"
                    profit = trade.get('profit', 0)
                    profit_text = f"✅ 盈利 {profit:.2f}%" if profit > 0 else f"❌ 亏损 {abs(profit):.2f}%"
                    analysis.append(f"• {trade.get('date', '未知')} {trade_type}: {profit_text}\n")
            
            return "".join(analysis)
        except Exception as e:
            return f"❌ 回测分析出错：{str(e)}"
    
    def _generate_recommendations(self, data, signals, risk_analysis, backtest_results):
        """生成交易建议"""
        try:
            recommendations = []
            
            # 获取最新信号
            latest_signal = signals[-1] if len(signals) > 0 else 0
            
            # 基于技术指标的建议
            rsi = data['RSI'].iloc[-1]
            ma_10 = data['MA_10'].iloc[-1]
            ma_50 = data['MA_50'].iloc[-1]
            volatility = data['Historical_Volatility'].iloc[-1]
            
            # 信号一致性分析
            signal_consistency = []
            
            # RSI信号
            if rsi > 70:
                signal_consistency.append(-1)  # 卖出信号
            elif rsi < 30:
                signal_consistency.append(1)   # 买入信号
                
            # 均线信号
            if ma_10 > ma_50:
                signal_consistency.append(1)
            elif ma_10 < ma_50:
                signal_consistency.append(-1)
                
            # 计算信号一致性
            if signal_consistency:
                avg_signal = sum(signal_consistency) / len(signal_consistency)
            else:
                avg_signal = 0
                
            # 生成建议
            if latest_signal == 1:
                recommendations.append("🎯 当前信号：建议买入\n")
                if avg_signal > 0:
                    recommendations.append("✅ 技术指标确认买入信号\n")
                else:
                    recommendations.append("⚠️ 技术指标显示存在不确定性，建议谨慎操作\n")
            elif latest_signal == -1:
                recommendations.append("🎯 当前信号：建议卖出\n")
                if avg_signal < 0:
                    recommendations.append("✅ 技术指标确认卖出信号\n")
                else:
                    recommendations.append("⚠️ 技术指标显示存在不确定性，建议谨慎操作\n")
            else:
                recommendations.append("🎯 当前信号：建议观望\n")
                
            # 风险提示
            if volatility > 20:
                recommendations.append("\n⚠️ 风险提示：\n")
                recommendations.append("• 市场波动性较高，建议降低交易规模\n")
            
            # 基于回测结果的建议
            if backtest_results and backtest_results.get('success'):
                metrics = backtest_results.get('data', {}).get('metrics', {})
                if metrics.get('win_rate', 0) < 50:
                    recommendations.append("\n💡 策略建议：\n")
                    recommendations.append("• 当前策略胜率偏低，建议调整交易参数\n")
                if metrics.get('max_drawdown', 0) > 20:
                    recommendations.append("• 策略最大回撤较大，建议设置合适的止损位\n")
            
            # 基于风险分析的建议
            if risk_analysis and risk_analysis.get('success'):
                portfolio_risk = risk_analysis.get('data', {}).get('portfolio_risk', {})
                if portfolio_risk.get('risk_level') == '高风险':
                    recommendations.append("\n⚠️ 组合风险提示：\n")
                    recommendations.append("• 当前组合风险较高，建议适当调整持仓结构\n")
            
            return "".join(recommendations)
        except Exception as e:
            return f"❌ 生成建议时出错：{str(e)}"