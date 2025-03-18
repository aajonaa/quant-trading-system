import requests
import textwrap

# DeepSeek API 配置
api_url = "https://api.deepseek.com/v1/chat/completions"
api_key = "sk-0252c7755aab4ba4aaa06bb2575ff187"  # 注意：实际使用请替换为有效密钥

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}


# 核心功能函数
def getanswers(_story, _instruct, _model="deepseek-chat"):
    """获取普通文本格式的响应"""
    payload = {
        "model": _model,
        "temperature": 0,
        "max_tokens": 2000,
        "messages": [
            {"role": "system", "content": _instruct},
            {"role": "user", "content": _story}
        ]
    }
    response = requests.post(api_url, headers=headers, json=payload)
    return response.json() if response.status_code == 200 else None


def getjson(_story, _instruct, _model="deepseek-chat"):
    """获取JSON格式的响应"""
    payload = {
        "model": _model,
        "temperature": 0,
        "max_tokens": 2000,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": f"{_instruct} 请确保响应为JSON格式"},
            {"role": "user", "content": _story}
        ]
    }
    response = requests.post(api_url, headers=headers, json=payload)
    return response.json() if response.status_code == 200 else None


# 主执行函数
def main():
    """示例分析数据（实际使用时替换为真实数据）"""
    analysis_data = """
    {
      "success": true,
      "explanation": {
        "USD/EUR": {
          "signal": "BUY",
          "confidence": 0.85,
          "risk_analysis": "Low Risk",
          "backtest_results": {
            "sharpe_ratio": 1.2,
            "drawdown": 5.6,
            "profit_factor": 1.8
          }
        },
        "USD/JPY": {
          "signal": "SELL",
          "confidence": 0.7,
          "risk_analysis": "Medium Risk",
          "backtest_results": {
            "sharpe_ratio": 0.95,
            "drawdown": 6.3,
            "profit_factor": 1.5
          }
        },
        "GBP/USD": {
          "signal": "HOLD",
          "confidence": 0.6,
          "risk_analysis": "High Risk",
          "backtest_results": {
            "sharpe_ratio": 0.8,
            "drawdown": 7.2,
            "profit_factor": 1.2
          }
        }
      }
    }

    {
      "success": true,
      "risk_report": {
        "USD/EUR": {
          "volatility": 0.12,
          "correlation": 0.75,
          "beta": 1.1
        },
        "USD/JPY": {
          "volatility": 0.09,
          "correlation": 0.65,
          "beta": 0.9
        },
        "GBP/USD": {
          "volatility": 0.11,
          "correlation": 0.8,
          "beta": 1.2
        }
      },
      "portfolio_risk": {
        "total_volatility": 0.1,
        "total_sharpe_ratio": 1.1,
        "max_drawdown": 4.5
      }
    }
    """

    instruction = """
    Please analyze and provide in-depth insights into the given currency pair’s trade signals, risk analysis, and backtest results. The goal is to interpret the data, explain the underlying trends, and provide actionable recommendations. For each currency pair, focus on the following aspects:

    1. Trade Signals:
    BUY Signal: Recommend entering the market at the given price level. Provide an explanation for the signal’s high confidence, discussing the broader trend and economic factors that support the bullish signal. Consider factors like strength of trend, economic data, and market conditions.
    SELL Signal: Recommend exiting the market at the given price level. Explain why this signal suggests a trend reversal or a good time to take profits. Include economic or technical factors contributing to the bearish outlook.
    HOLD Signal: Recommend maintaining the current position. Explain the lack of a clear trend and why it is better to wait. Describe the conditions that need to be met (e.g., breakout, change in market sentiment) before making a move.
    Stop-Loss Signal: Provide a detailed explanation of where to set the stop-loss, based on the pair’s volatility and risk management. Discuss the potential risk of adverse price movements and why the chosen stop-loss level minimizes this risk.
    Take Profit Signal: Provide suggestions for setting the take-profit level. Explain why the expected price move justifies this level, considering both technical and fundamental analysis.

    2. Risk Analysis: For each currency pair, offer a detailed risk assessment by evaluating the following:
    Volatility: Discuss the anticipated price movement range (measured by standard deviations of returns). Highlight how the pair’s volatility impacts the likelihood of large price swings and the risk involved.
    Correlation: Explain how the currency pair’s performance correlates with broader market trends or other assets. A higher correlation means more sensitivity to external factors like global market movements or risk appetite.
    Beta: Analyze the pair's sensitivity to broader market movements, with a focus on beta values. A beta above 1 suggests greater volatility relative to the market, while a beta below 1 indicates that the pair tends to move less aggressively than the market.
    Market Conditions: Identify key external factors influencing the currency pair, such as upcoming economic reports, central bank decisions, or geopolitical events. Discuss how these conditions might affect the pair’s behavior in the short to medium term.
    Backtest Results:

    3. Expected Profit: Explain the expected profit based on backtest results. Discuss how this aligns with the volatility and strength of the trade signal. Include considerations for both short-term and long-term expectations.
    Max Drawdown: Analyze the maximum drawdown during the backtest period. Discuss what this tells us about potential risk in real-market conditions. How much loss is tolerable based on the strategy’s risk profile?
    Winning Rate: Assess the historical success rate of the strategy, analyzing the ratio of winning to losing trades. Discuss whether the strategy’s consistency supports reliable returns over time or if improvements are needed.
    Return Rate: Review the overall return based on backtesting. How does the return compare to the amount of risk taken, as indicated by the Sharpe ratio? Discuss whether the returns justify the risk involved in each pair’s strategy.

    """

    response = getjson(str(analysis_data), instruction)

    if response and "choices" in response:
        result = response["choices"][0]["message"]["content"]
        print("结构化分析结果：\n", textwrap.fill(result, width=100))
    else:
        print("分析请求失败")


if __name__ == "__main__":
    main()
