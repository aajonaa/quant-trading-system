from flask import Flask, render_template, request, jsonify, session
from backend.backtest.backtest_optimum import BacktestOptimizer
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


@app.route('/api/optimize_strategy', methods=['POST'])
def optimize_strategy():
    try:
        data = request.get_json()
        optimizer = BacktestOptimizer()

        results = optimizer.optimize(
            currency_pair=data['currency_pair'],
            start_date=data['start_date'],
            end_date=data['end_date'],
            population_size=data.get('population_size', 50),
            generations=data.get('generations', 30)
        )

        return jsonify(results)

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

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

@app.route('/api/single_backtest', methods=['POST'])
def single_backtest():
    try:
        data = request.get_json()
        currency_pair = data.get('currency_pair')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        # 创建回测器实例
        backtester = ForexBacktester()
        
        # 执行单一货币对回测
        results = backtester.run_backtest(
            currency_pair=currency_pair,
            start_date=start_date,
            end_date=end_date
        )
        
        return jsonify({
            'success': True,
            'results': {
                'total_return': results['total_return'],
                'sharpe_ratio': results['sharpe_ratio'],
                'max_drawdown': results['max_drawdown'],
                'win_rate': results['win_rate'],
                'trades': results['trades'],
                'equity_curve': results['equity_curve'].tolist()
            }
        })
    except Exception as e:
        logger.error(f"单一货币对回测错误: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/multi_currency_risk', methods=['POST'])
def multi_currency_risk():
    try:
        data = request.get_json()
        selected_pairs = data.get('currency_pairs', [])
        
        # 从CSV文件读取风险分析结果
        risk_data = pd.read_csv('backend/mulsignals/currency_pair_risks.csv')
        
        # 过滤选中的货币对
        filtered_data = risk_data[risk_data['货币对组合'].isin(selected_pairs)]
        
        return jsonify({
            'success': True,
            'risk_analysis': filtered_data.to_dict('records')
        })
    except Exception as e:
        logger.error(f"多货币风险分析错误: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
