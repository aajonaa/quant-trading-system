from flask import Flask, render_template, jsonify, request
import pandas as pd
from datetime import datetime
import numpy as np
import logging
from backend.utils import setup_logging
from backend.explainer.explainer import SignalExplainer
import os
from backend.config import Config
from backend.model_analysis import (
    ForexModelAnalyzer,
    MultiCurrencyRiskAnalyzer,
)
from backend.backtest.optimization import BacktestOptimizer
from backend.backtest.backtest import Backtester

app = Flask(__name__)

# 初始化日志
setup_logging()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/load_data', methods=['POST'])
def load_data():
    try:
        currency_pair = request.json.get('currency_pair')
        if not currency_pair:
            return jsonify({'error': '请选择货币对'}), 400

        # 确保数据目录存在
        os.makedirs(Config.DATA_DIR, exist_ok=True)

        # 使用ForexDataGenerator获取数据
        forex_generator = ForexDataGenerator()
        data_list, csv_path = forex_generator.get_forex_data(
            currency_pair,
            period="6mo",
            interval="1d",
            save_csv=True
        )

        if not data_list:
            return jsonify({'error': f'没有找到 {currency_pair} 的数据'}), 404

        # 记录成功信息
        logging.info(f"成功加载 {currency_pair} 的数据，共 {len(data_list)} 条记录")

        # 将Path对象转换为字符串
        csv_path_str = str(csv_path) if csv_path else None

        return jsonify({
            'success': True,
            'data': data_list,
            'csv_path': csv_path_str  # 使用字符串而不是Path对象
        })

    except Exception as e:
        logging.error(f"加载数据时出错: {str(e)}")
        return jsonify({'error': f'加载数据时出错: {str(e)}'}), 500

@app.route('/generate_signals', methods=['POST'])
def generate_signals():
    """生成交易信号"""
    try:
        # 获取请求数据
        data = request.get_json()
        currency_pair = data.get('currency_pair')

        if not currency_pair:
            return jsonify({'success': False, 'error': '未提供货币对'}), 400

        # 获取外汇数据
        data_generator = ForexDataGenerator()
        data_list, _ = data_generator.get_forex_data(currency_pair)

        if not data_list or len(data_list) == 0:
            return jsonify({'success': False, 'error': '获取外汇数据失败'}), 500

        # 转换为DataFrame
        signals_df = pd.DataFrame(data_list)

        if signals_df.empty:
            return jsonify({'success': False, 'error': '数据转换失败'}), 500

        # 确保必要的列存在
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in signals_df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in signals_df.columns]
            return jsonify({'success': False, 'error': f'数据缺少必要的列: {missing_cols}'}), 500

        # 设置日期索引
        signals_df['Date'] = pd.to_datetime(signals_df['Date'])
        signals_df.set_index('Date', inplace=True)

        # 确保数值类型正确
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            signals_df[col] = pd.to_numeric(signals_df[col], errors='coerce')

        # 处理NaN值
        signals_df = signals_df.ffill().bfill()
        signals_df = signals_df.fillna(0)

        # 初始化模型分析器
        model_analyzer = ForexModelAnalyzer()

        # 生成信号
        signals = model_analyzer.generate_ensemble_signals(signals_df, currency_pair)

        if signals is None:
            return jsonify({'success': False, 'error': '生成信号失败'}), 500

        # 将信号添加到DataFrame
        signals_df['Signal'] = signals

        # 生成风险分析数据
        risk_analyzer = MultiCurrencyRiskAnalyzer()

        # 创建包含多个货币对的数据字典
        currency_pairs = ['USD/EUR', 'USD/JPY', 'GBP/USD', 'AUD/USD', 'USD/CHF', 'NZD/USD', 'USD/CAD']
        data_dict = {}
        for pair in currency_pairs:
            pair_data, _ = data_generator.get_forex_data(pair)
            if pair_data:
                data_dict[pair] = pd.DataFrame(pair_data)

        # 生成风险信号
        risk_analysis = risk_analyzer.generate_risk_signals(data_dict)

        # 确保所有数值都是有效的JSON格式
        def clean_numeric(x):
            if pd.isna(x):
                return 0
            if isinstance(x, (np.int64, np.int32)):
                return int(x)
            if isinstance(x, (np.float64, np.float32)):
                return float(x)
            return x

        # 准备返回数据
        result_data = signals_df.reset_index().to_dict('records')

        # 清理数据中的无效值
        for record in result_data:
            for key, value in record.items():
                record[key] = clean_numeric(value)
                if isinstance(record[key], datetime):
                    record[key] = record[key].strftime('%Y-%m-%d')

        # 获取模型准确度信息
        accuracies = model_analyzer.get_model_accuracies(currency_pair) if hasattr(model_analyzer, 'get_model_accuracies') else None

        # 如果有准确度信息，确保其中的数值也是有效的JSON格式
        if accuracies:
            for key, value in accuracies.items():
                if isinstance(value, dict):
                    accuracies[key] = {k: clean_numeric(v) for k, v in value.items()}
                else:
                    accuracies[key] = clean_numeric(value)

        return jsonify({
            'success': True,
            'data': result_data,
            'accuracies': accuracies,
            'risk_analysis': risk_analysis
        })

    except Exception as e:
        logging.error(f"生成信号时出错: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# 辅助函数
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.inf)
    return 100 - (100 / (1 + rs))

def calculate_atr(df, period=14):
    tr = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        )
    )
    return tr.rolling(window=period).mean()

def calculate_volatility(df, period=20):
    return df['Close'].pct_change().rolling(window=period).std() * np.sqrt(252) * 100

def load_forex_data(currency_pair):
    """
    加载外汇数据

    Args:
        currency_pair (str): 货币对名称

    Returns:
        pd.DataFrame: 处理后的外汇数据
    """
    try:
        # 使用ForexDataGenerator获取数据
        data_generator = ForexDataGenerator()
        data_list, _ = data_generator.get_forex_data(currency_pair)

        if not data_list or len(data_list) == 0:
            return None

        # 转换为DataFrame
        df = pd.DataFrame(data_list)

        # 设置日期索引
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        # 确保数值类型正确
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 计算技术指标
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        df['RSI'] = calculate_rsi(df['Close'])
        df['ATR'] = calculate_atr(df)
        df['Historical_Volatility'] = calculate_volatility(df)

        # 处理缺失值
        df = df.ffill().bfill()
        df = df.fillna(0)

        return df

    except Exception as e:
        logging.error(f"加载外汇数据时出错: {str(e)}")
        return None

@app.route('/explain_signal', methods=['POST'])
def explain_signal():
    try:
        data = request.get_json()
        currency_pair = data.get('currency_pair')

        if not currency_pair:
            return jsonify({'success': False, 'error': '缺少货币对参数'})

        # 获取历史数据
        df = load_forex_data(currency_pair)
        if df is None or df.empty:
            return jsonify({'success': False, 'error': '无法获取历史数据'})

        # 生成信号
        analyzer = ForexModelAnalyzer()
        signals = analyzer.generate_ensemble_signals(df, currency_pair)

        if signals is None:
            return jsonify({'success': False, 'error': '生成信号失败'})

        # 获取风险分析
        risk_analyzer = MultiCurrencyRiskAnalyzer()

        # 创建包含多个货币对的数据字典
        currency_pairs = ['USD/EUR', 'USD/JPY', 'GBP/USD', 'AUD/USD', 'USD/CHF', 'NZD/USD', 'USD/CAD']
        data_dict = {}
        for pair in currency_pairs:
            pair_df = load_forex_data(pair)
            if pair_df is not None:
                data_dict[pair] = pair_df

        # 生成风险分析
        risk_analysis = risk_analyzer.generate_risk_signals(data_dict)

        # 运行回测
        backtester = Backtester()
        backtest_results = backtester.run_backtest(df, signals, {
            'stop_loss': 0.02,
            'take_profit': 0.03,
            'position_size': 1000,
            'initial_capital': 100000
        })

        if not backtest_results['success']:
            return jsonify({'success': False, 'error': '回测失败'})

        # 生成解释
        explainer = SignalExplainer()
        explanation = explainer.explain_signals(
            data=df,
            signals=signals,
            risk_analysis=risk_analysis,
            backtest_results=backtest_results
        )

        return jsonify({
            'success': True,
            'explanation': explanation
        })

    except Exception as e:
        logging.error(f"生成解释时出错: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'生成解释时出错: {str(e)}'
        })

@app.route('/analyze_portfolio_risk', methods=['POST'])
def analyze_portfolio_risk():
    try:
        # 获取所有货币对的数据
        forex_generator = ForexDataGenerator()
        data_dict = {}

        for pair in ['USD/EUR', 'USD/JPY', 'GBP/USD', 'AUD/USD',
                    'USD/CHF', 'NZD/USD', 'USD/CAD']:
            data_list, _ = forex_generator.get_forex_data(pair, save_csv=False)
            data_dict[pair] = pd.DataFrame(data_list)

        # 创建风险分析器实例
        risk_analyzer = MultiCurrencyRiskAnalyzer()

        # 生成风险信号
        risk_report = risk_analyzer.generate_risk_signals(data_dict)

        # 计算示例投资组合风险
        portfolio_weights = {pair: 1/len(data_dict) for pair in data_dict}  # 等权重
        portfolio_risk = risk_analyzer.calculate_portfolio_risk(data_dict, portfolio_weights)

        # 合并结果
        result = {
            'success': True,
            'risk_report': risk_report,
            'portfolio_risk': portfolio_risk
        }

        return jsonify(result)

    except Exception as e:
        logging.error(f"分析投资组合风险时出错: {str(e)}")
        return jsonify({'error': f'分析投资组合风险时出错: {str(e)}'}), 500

@app.route('/run_backtest', methods=['POST'])
def run_backtest():
    """运行回测"""
    try:
        # 获取请求数据
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': '请求数据为空'}), 400

        currency_pair = data.get('currency_pair')
        if not currency_pair:
            return jsonify({'success': False, 'error': '未提供货币对'}), 400

        # 获取回测参数
        params = data.get('params', {})

        # 生成数据
        data_generator = ForexDataGenerator()
        data_list, _ = data_generator.get_forex_data(currency_pair)

        if not data_list or len(data_list) == 0:
            return jsonify({'success': False, 'error': '无法获取外汇数据'}), 400

        # 转换为DataFrame并确保所有必要的列都存在
        df = pd.DataFrame(data_list)
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            return jsonify({'success': False, 'error': f'数据缺少必要的列: {missing_cols}'}), 400

        # 确保日期列格式正确
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        # 确保数据类型正确并处理无效值
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 使用前向填充和后向填充处理缺失值
        df = df.ffill().bfill()

        # 计算技术指标
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        df['RSI'] = calculate_rsi(df['Close'])
        df['ATR'] = calculate_atr(df)
        df['Historical_Volatility'] = calculate_volatility(df)

        # 再次处理可能的NaN值
        df = df.ffill().bfill()
        df = df.fillna(0)  # 处理任何剩余的NaN值

        # 生成交易信号
        model_analyzer = ForexModelAnalyzer()
        signals = model_analyzer.generate_ensemble_signals(df.copy(), currency_pair)

        if signals is None:
            return jsonify({'success': False, 'error': '生成信号失败'}), 500

        # 设置默认参数
        default_params = {
            'stop_loss': 0.02,
            'take_profit': 0.03,
            'position_size': 1000,
            'initial_capital': 100000,
            'commission': 0.0002,  # 手续费率
            'slippage': 0.0001    # 滑点率
        }

        # 更新参数
        default_params.update(params)

        # 运行回测
        backtester = Backtester()
        results = backtester.run_backtest(df.copy(), signals, default_params)

        if not results or not results.get('success'):
            error_msg = results.get('error', '回测失败') if results else '回测失败'
            return jsonify({'success': False, 'error': error_msg}), 500

        # 处理日期格式
        if 'data' in results and 'trades' in results['data']:
            for trade in results['data']['trades']:
                if isinstance(trade['date'], (pd.Timestamp, datetime)):
                    trade['date'] = trade['date'].strftime('%Y-%m-%d')

        return jsonify(results)

    except Exception as e:
        logging.error(f"运行回测时出错: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/optimize_backtest', methods=['POST'])
def optimize_backtest():
    """优化回测参数"""
    try:
        # 获取请求数据
        data = request.get_json()
        if not data or 'currency_pair' not in data:
            return jsonify({'error': '缺少必要的参数'}), 400

        currency_pair = data['currency_pair']
        optimization_method = data.get('method', 'grid')  # 默认使用网格搜索

        # 获取历史数据
        data_generator = ForexDataGenerator()
        data_list, _ = data_generator.get_forex_data(currency_pair)

        if not data_list or len(data_list) == 0:
            return jsonify({'error': '无法获取历史数据'}), 400

        # 转换为DataFrame
        df = pd.DataFrame(data_list)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        # 创建回测器和优化器实例
        backtester = Backtester()

        # 定义参数网格
        param_grid = {
            'stop_loss': [0.01, 0.02, 0.03],
            'take_profit': [0.02, 0.03, 0.04],
            'position_size': [1000, 2000, 5000],
            'initial_capital': [100000],
            'commission': [0.0002],
            'slippage': [0.0001]
        }

        # 运行优化
        analyzer = ForexModelAnalyzer()
        strategy_func = lambda df, params: analyzer.generate_ensemble_signals(df, currency_pair)

        results = backtester.optimize_parameters(df, strategy_func, param_grid)

        if not results['success']:
            return jsonify({'success': False, 'error': results['error']}), 500

        # 使用最佳参数运行回测
        best_params = results['data']['best_params']
        signals = analyzer.generate_ensemble_signals(df, currency_pair)
        final_result = backtester.run_backtest(df, signals, best_params)

        if not final_result['success']:
            return jsonify({'success': False, 'error': final_result['error']}), 500

        # 处理日期格式
        if 'data' in final_result and 'trades' in final_result['data']:
            for trade in final_result['data']['trades']:
                if isinstance(trade['date'], (pd.Timestamp, datetime)):
                    trade['date'] = trade['date'].strftime('%Y-%m-%d')

        # 构建响应数据
        response_data = {
            'success': True,
            'data': {
                'optimization_results': {
                    'best_params': best_params,
                    'best_metrics': results['data']['best_metrics']
                },
                'backtest_results': final_result['data']
            }
        }

        return jsonify(response_data)

    except Exception as e:
        logging.error(f"优化回测时出错: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/run_optimized_backtest', methods=['POST'])
def run_optimized_backtest():
    """运行优化后的回测"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': '无效的请求数据'
            })

        currency_pair = data.get('currency_pair')
        optimization_method = data.get('optimization_method')

        if not currency_pair or not optimization_method:
            return jsonify({
                'success': False,
                'error': '缺少必要的参数'
            })

        # 加载历史数据
        df = load_forex_data(currency_pair)
        if df is None or df.empty:
            return jsonify({
                'success': False,
                'error': '无法加载历史数据'
            })

        # 创建优化器实例
        optimizer = BacktestOptimizer()

        # 设置优化参数范围
        optimization_params = {
            'stop_loss_range': [0.01, 0.02, 0.03],
            'take_profit_range': [0.02, 0.03, 0.04],
            'position_size_range': [1000, 2000, 5000],
            'initial_capital': 100000,
            'commission': 0.0002,
            'slippage': 0.0001
        }

        # 根据不同的优化方法设置特定参数
        if optimization_method == 'random_search':
            optimization_params['n_iterations'] = 20
        elif optimization_method == 'bayesian_optimization':
            optimization_params['n_iterations'] = 15
        elif optimization_method == 'genetic_algorithm':
            optimization_params['population_size'] = 20
            optimization_params['n_generations'] = 10

        # 运行优化
        optimization_result = optimizer.optimize(df, currency_pair, optimization_method, optimization_params)

        if not optimization_result['success']:
            return jsonify(optimization_result)

        # 使用优化后的参数运行回测
        backtester = Backtester()

        # 生成交易信号
        analyzer = ForexModelAnalyzer()
        signals = analyzer.generate_ensemble_signals(df, currency_pair)

        if signals is None:
            return jsonify({
                'success': False,
                'error': '生成交易信号失败'
            })

        # 运行回测
        backtest_result = backtester.run_backtest(
            df,
            signals,
            optimization_result['data']['optimization_results']['best_params']
        )

        if not backtest_result['success']:
            return jsonify({
                'success': False,
                'error': '使用优化参数运行回测失败'
            })

        # 合并优化结果和回测结果
        result = {
            'success': True,
            'data': {
                'metrics': backtest_result['data']['metrics'],
                'trades': backtest_result['data']['trades'],
                'optimization_results': optimization_result['data']['optimization_results']
            }
        }

        # 处理日期格式
        if 'trades' in result['data']:
            for trade in result['data']['trades']:
                if isinstance(trade['date'], (pd.Timestamp, datetime)):
                    trade['date'] = trade['date'].strftime('%Y-%m-%d')

        return jsonify(result)

    except Exception as e:
        app.logger.error(f"运行优化回测时出错: {str(e)}")
        return jsonify({
            'success': False,
            'error': f"运行优化回测时出错: {str(e)}"
        })

if __name__ == '__main__':
    app.run(debug=True)
