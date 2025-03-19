import requests
import textwrap
import pandas as pd
import json
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# DeepSeek API 配置
api_url = "https://api.deepseek.com/v1/chat/completions"
api_key = "sk-0252c7755aab4ba4aaa06bb2575ff187"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

def load_signal_data(currency_pair):
    """加载货币对的信号数据和处理后的特征数据"""
    try:
        base_dir = Path(__file__).parent.parent
        
        # 加载信号数据
        signals_path = base_dir / 'signals' / f"{currency_pair}_signals.csv"
        if not signals_path.exists():
            logger.error(f"信号文件不存在: {signals_path}")
            return None, None
            
        signals_df = pd.read_csv(signals_path)
        
        # 加载处理后的特征数据
        processed_path = base_dir / 'FE' / f"{currency_pair}_processed.csv"
        if not processed_path.exists():
            logger.error(f"处理后的特征文件不存在: {processed_path}")
            return signals_df, None
            
        processed_df = pd.read_csv(processed_path)
        
        return signals_df, processed_df
        
    except Exception as e:
        logger.error(f"加载数据出错: {str(e)}")
        return None, None

def combine_data(signals_df, processed_df):
    """合并信号数据和处理后的特征数据"""
    if signals_df is None:
        return None
        
    # 如果没有处理后的特征数据，只返回信号数据
    if processed_df is None:
        return signals_df
    
    try:
        # 确保两个数据集有相同的日期列
        signals_df['Date'] = pd.to_datetime(signals_df['Date'])
        processed_df['Date'] = pd.to_datetime(processed_df['Date'])
        
        # 合并数据集
        combined_df = pd.merge(signals_df, processed_df, on='Date', how='inner')
        return combined_df
        
    except Exception as e:
        logger.error(f"合并数据出错: {str(e)}")
        return signals_df

def prepare_analysis_data(currency_pair):
    """准备用于分析的数据"""
    signals_df, processed_df = load_signal_data(currency_pair)
    if signals_df is None:
        return None
        
    combined_df = combine_data(signals_df, processed_df)
    if combined_df is None:
        return None
    
    # 提取最近的数据点进行分析（最多30行）
    recent_data = combined_df.tail(30)
    
    # 计算一些基本统计信息
    stats = {
        "currency_pair": currency_pair,
        "period": f"{recent_data['Date'].min().strftime('%Y-%m-%d')} 至 {recent_data['Date'].max().strftime('%Y-%m-%d')}",
        "avg_price": recent_data['Price'].mean() if 'Price' in recent_data.columns else None,
        "price_change": recent_data['Price'].pct_change().mean() * 100 if 'Price' in recent_data.columns else None,
        "signal_distribution": recent_data['Signal'].value_counts().to_dict() if 'Signal' in recent_data.columns else None,
        "recent_signals": recent_data[['Date', 'Signal', 'Price']].tail(10).to_dict('records') if all(col in recent_data.columns for col in ['Date', 'Signal', 'Price']) else None
    }
    
    # 添加技术指标（如果存在）
    technical_indicators = {}
    for col in recent_data.columns:
        if any(indicator in col for indicator in ['SMA', 'EMA', 'RSI', 'MACD', 'BB_']):
            technical_indicators[col] = recent_data[col].iloc[-1]
    
    stats["technical_indicators"] = technical_indicators
    
    return stats

def generate_signal_explanation(currency_pair):
    """生成货币对信号的专业解释"""
    analysis_data = prepare_analysis_data(currency_pair)
    if not analysis_data:
        return {"success": False, "error": f"无法获取 {currency_pair} 的数据"}
    
    # 构建提示词
    prompt = f"""
    请分析以下货币对 {currency_pair} 的交易信号和技术指标数据，并提供专业的市场分析报告。
    
    数据概览:
    - 货币对: {currency_pair}
    - 分析周期: {analysis_data['period']}
    - 平均价格: {analysis_data['avg_price']}
    - 价格变化率: {analysis_data['price_change']}%
    
    信号分布:
    {json.dumps(analysis_data['signal_distribution'], indent=2)}
    
    最近信号:
    {json.dumps(analysis_data['recent_signals'], indent=2)}
    
    技术指标:
    {json.dumps(analysis_data['technical_indicators'], indent=2)}
    
    请提供以下内容的专业分析:
    1. 市场趋势概述：基于价格走势和技术指标分析当前市场趋势
    2. 信号解读：解释最近的交易信号含义及其可靠性
    3. 技术面分析：详细分析各项技术指标的意义和相互验证情况
    4. 交易建议：提供具体的入场点、止损位和目标价位
    5. 风险评估：评估当前市场风险水平和潜在的风险因素
    6. 未来展望：预测短期内可能的价格走势和关键支撑/阻力位
    
    请以专业金融分析师的口吻撰写，内容要详实、专业且有实用价值。
    """
    
    # 调用 DeepSeek API
    try:
        payload = {
            "model": "deepseek-chat",
            "temperature": 0.2,
            "max_tokens": 2000,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": "你是一位专业的外汇市场分析师，擅长解读交易信号和技术指标，提供专业的市场分析和交易建议。请确保响应为JSON格式。"},
                {"role": "user", "content": prompt}
            ]
        }
        
        response = requests.post(api_url, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                explanation = json.loads(result["choices"][0]["message"]["content"])
                return {
                    "success": True,
                    "currency_pair": currency_pair,
                    "explanation": explanation
                }
            else:
                logger.error(f"API响应格式错误: {result}")
                return {"success": False, "error": "API响应格式错误"}
        else:
            logger.error(f"API请求失败: {response.status_code} - {response.text}")
            return {"success": False, "error": f"API请求失败: {response.status_code}"}
            
    except Exception as e:
        logger.error(f"生成解释时出错: {str(e)}")
        return {"success": False, "error": str(e)}

def get_signal_explanation(currency_pair):
    """获取货币对信号的专业解释（对外接口）"""
    return generate_signal_explanation(currency_pair)

# 测试函数
def main():
    """测试函数"""
    currency_pairs = ["CNYAUD", "CNYEUR", "CNYGBP", "CNYJPY", "CNYUSD"]
    
    for pair in currency_pairs:
        print(f"\n正在分析 {pair}...")
        result = get_signal_explanation(pair)
        
        if result["success"]:
            print(f"{pair} 分析成功!")
            print("分析报告摘要:")
            explanation = result["explanation"]
            if "市场趋势概述" in explanation:
                print(f"- 市场趋势: {explanation['市场趋势概述'][:1000]}...")
            if "交易建议" in explanation:
                print(f"- 交易建议: {explanation['交易建议'][:1000]}...")
        else:
            print(f"{pair} 分析失败: {result['error']}")

if __name__ == "__main__":
    main()
