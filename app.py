from flask import Flask, render_template, request, jsonify, session

from pathlib import Path
import logging
import sys
import pandas as pd

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
backend_dir = current_dir / 'backend'
sys.path.append(str(current_dir))
sys.path.append(str(backend_dir))  # 添加backend目录到路径

# 导入自定义模块
from backend.backtest.backtester import ForexBacktester
from backend.mulcurrency_risk import MultiCurrencyRiskAnalyzer
from backend.model_analysis import MultiStepPredictor

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # 用于session加密

# 模拟用户数据库
users_db = {
    'admin': {
        'password': 'admin123',
        'email': 'admin@example.com'
    }
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/backtest', methods=['POST'])
def backtest():
    try:
        # 获取请求参数
        currency = request.form.get('currency')
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        
        if not currency or not start_date or not end_date:
            return jsonify({'error': '缺少必要参数'})
        
        # 创建回测器实例
        backtester = ForexBacktester()
        
        # 加载数据
        if not backtester.load_data():
            return jsonify({'error': '加载数据失败'})
        
        # 设置回测参数
        params = {
            'start_date': start_date,
            'end_date': end_date
        }
        
        # 执行回测
        if not backtester.run_backtest(params):
            return jsonify({'error': '回测执行失败'})
        
        # 获取回测结果
        if currency not in backtester.backtest_results:
            return jsonify({'error': f'未找到 {currency} 的回测结果'})
        
        result = backtester.backtest_results[currency]
        
        # 准备返回数据
        equity_curve = result['equity_curve']
        
        # 转换日期格式
        dates = [d.strftime('%Y-%m-%d') for d in equity_curve.index]
        values = equity_curve['capital'].tolist()
        
        # 获取价格和信号数据
        price_data = backtester.pairs_data[currency]['Price'].loc[start_date:end_date].tolist()
        signals = backtester.pairs_data[currency]['Signal'].loc[start_date:end_date].tolist()
        
        # 返回结果
        return jsonify({
            'currency': currency,
            'metrics': {
                'total_return': result['total_return'],
                'annual_return': result['annual_return'],
                'sharpe_ratio': result['sharpe_ratio'],
                'sortino_ratio': result['sortino_ratio'],
                'max_drawdown': result['max_drawdown'],
                'win_rate': result['win_rate'],
                'profit_factor': result['profit_factor'],
                'avg_trade': result['avg_trade'],
                'max_consecutive_losses': result['max_consecutive_losses']
            },
            'equity_curve': {
                'dates': dates,
                'values': values
            },
            'price_data': price_data,
            'signals': signals,
            'start_date': start_date,
            'end_date': end_date
        })
        
    except Exception as e:
        logger.error(f"回测API错误: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': f'服务器错误: {str(e)}'})

@app.route('/api/risk_analysis', methods=['POST'])
def risk_analysis():
    try:
        # 直接读取mulsignals中的结果
        risk_data = pd.read_csv('backend/mulsignals/currency_pair_risks.csv')
        
        return jsonify({
            'pair_risks': risk_data.to_dict('records')
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/optimize', methods=['POST'])
def optimize():
    try:
        # 获取请求参数
        currency = request.form.get('currency')
        method = request.form.get('method')
        target = request.form.get('target')
        stop_loss = float(request.form.get('stop_loss', 0.02))
        take_profit = float(request.form.get('take_profit', 0.04))
        position_size = float(request.form.get('position_size', 0.5))
        trailing_stop = float(request.form.get('trailing_stop', 0.015))
        
        if not currency:
            return jsonify({'error': '请选择货币对'})
        
        # 创建回测器实例
        backtester = ForexBacktester()
        
        # 加载数据
        if not backtester.load_data():
            return jsonify({'error': '加载数据失败'})
        
        # 执行原始回测
        original_params = {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'trailing_stop': trailing_stop
        }
        
        if not backtester.run_backtest(original_params):
            return jsonify({'error': '原始回测执行失败'})
        
        # 获取原始回测结果
        if currency not in backtester.backtest_results:
            return jsonify({'error': f'未找到 {currency} 的回测结果'})
        
        original_result = backtester.backtest_results[currency]
        
        # 模拟优化过程
        # 在实际应用中，这里应该实现真正的优化算法
        optimized_params = {
            'stop_loss': min(stop_loss * 0.8, 0.01),  # 减小止损
            'take_profit': take_profit * 1.2,  # 增加止盈
            'position_size': min(position_size * 1.1, 1.0),  # 增加仓位
            'trailing_stop': trailing_stop * 1.1  # 增加追踪止损
        }
        
        # 执行优化后的回测
        if not backtester.run_backtest(optimized_params):
            return jsonify({'error': '优化回测执行失败'})
        
        # 获取优化后的回测结果
        optimized_result = backtester.backtest_results[currency]
        
        # 准备返回数据
        # 转换日期格式
        original_dates = [d.strftime('%Y-%m-%d') for d in original_result['equity_curve'].index]
        original_values = original_result['equity_curve']['capital'].tolist()
        
        optimized_dates = [d.strftime('%Y-%m-%d') for d in optimized_result['equity_curve'].index]
        optimized_values = optimized_result['equity_curve']['capital'].tolist()
        
        # 返回结果
        return jsonify({
            'currency': currency,
            'method': method,
            'target': target,
            'best_params': optimized_params,
            'before': {
                'total_return': original_result['total_return'],
                'annual_return': original_result['annual_return'],
                'sharpe_ratio': original_result['sharpe_ratio'],
                'max_drawdown': original_result['max_drawdown'],
                'win_rate': original_result['win_rate']
            },
            'after': {
                'total_return': optimized_result['total_return'],
                'annual_return': optimized_result['annual_return'],
                'sharpe_ratio': optimized_result['sharpe_ratio'],
                'max_drawdown': optimized_result['max_drawdown'],
                'win_rate': optimized_result['win_rate']
            },
            'before_equity': {
                'dates': original_dates,
                'values': original_values
            },
            'after_equity': {
                'dates': optimized_dates,
                'values': optimized_values
            }
        })
        
    except Exception as e:
        logger.error(f"优化API错误: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': f'服务器错误: {str(e)}'})

@app.route('/api/apply_optimization', methods=['POST'])
def apply_optimization():
    try:
        # 获取请求参数
        currency = request.form.get('currency')
        stop_loss = float(request.form.get('stop_loss', 0.02))
        take_profit = float(request.form.get('take_profit', 0.04))
        position_size = float(request.form.get('position_size', 0.5))
        trailing_stop = float(request.form.get('trailing_stop', 0.015))
        
        if not currency:
            return jsonify({'error': '请选择货币对'})
        
        # 在实际应用中，这里应该将优化后的参数保存到数据库或配置文件中
        # 这里只是模拟成功响应
        
        return jsonify({
            'success': True,
            'message': '优化参数已成功应用',
            'currency': currency,
            'params': {
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_size': position_size,
                'trailing_stop': trailing_stop
            }
        })
        
    except Exception as e:
        logger.error(f"应用优化API错误: {str(e)}")
        return jsonify({'error': f'服务器错误: {str(e)}'})

@app.route('/api/login', methods=['POST'])
def login():
    try:
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            return jsonify({'error': '请提供用户名和密码'})
        
        # 检查用户是否存在
        if username not in users_db:
            return jsonify({'error': '用户名或密码错误'})
        
        # 检查密码是否正确
        if users_db[username]['password'] != password:
            return jsonify({'error': '用户名或密码错误'})
        
        # 登录成功，设置session
        session['logged_in'] = True
        session['username'] = username
        
        return jsonify({
            'success': True,
            'username': username
        })
        
    except Exception as e:
        logger.error(f"登录API错误: {str(e)}")
        return jsonify({'error': f'服务器错误: {str(e)}'})

@app.route('/api/register', methods=['POST'])
def register():
    try:
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if not username or not email or not password:
            return jsonify({'error': '请提供所有必填字段'})
        
        # 检查用户是否已存在
        if username in users_db:
            return jsonify({'error': '用户名已存在'})
        
        # 注册新用户
        users_db[username] = {
            'password': password,
            'email': email
        }
        
        return jsonify({
            'success': True,
            'message': '注册成功'
        })
        
    except Exception as e:
        logger.error(f"注册API错误: {str(e)}")
        return jsonify({'error': f'服务器错误: {str(e)}'})

@app.route('/api/logout', methods=['POST'])
def logout():
    try:
        # 清除session
        session.pop('logged_in', None)
        session.pop('username', None)
        
        return jsonify({
            'success': True,
            'message': '已退出登录'
        })
        
    except Exception as e:
        logger.error(f"退出登录API错误: {str(e)}")
        return jsonify({'error': f'服务器错误: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)
