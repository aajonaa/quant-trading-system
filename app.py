from flask import Flask, render_template, request, jsonify, session
from backend.backtest.backtest_optimum import StrategyOptimizer
from pathlib import Path
import logging
import sys
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash
import os
import psycopg2

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
backend_dir = current_dir / 'backend'
sys.path.append(str(current_dir))
sys.path.append(str(backend_dir))  # 添加backend目录到路径

# 导入自定义模块
from backend.backtest.backtester import ForexBacktester
from backend.mulcurrency_risk import MultiCurrencyRiskAnalyzer
from backend.model_analysis import MultiStepPredictor
from Database.Pgadmin4 import DatabaseManager

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Use environment variable for secret key
app.secret_key = os.environ.get('SECRET_KEY', 'your_secret_key')

# 模拟用户数据库
users_db = {
    'admin': {
        'password': 'admin123',
        'email': 'admin@example.com'
    }
}

class DatabaseManager:
    def __init__(self):
        self.conn = psycopg2.connect(
            host=os.environ.get('DB_HOST', 'localhost'),
            database=os.environ.get('DB_NAME', 'quant_trading_db'),
            user=os.environ.get('DB_USER', 'postgres'),
            password=os.environ.get('DB_PASSWORD', 'jonawong.'),
            port=os.environ.get('DB_PORT', 5432)
        )
        self.cur = self.conn.cursor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/backtest')
def backtest_page():
    return render_template('backtest.html')

@app.route('/signal')
def signal_page():
    return render_template('signal.html')

@app.route('/risk')
def risk_page():
    return render_template('risk.html')

@app.route('/optimize')
def optimize_page():
    return render_template('optimize.html')

@app.route('/api/optimize_strategy', methods=['POST'])
def optimize_strategy():
    try:
        data = request.get_json()
        currency_pair = data.get('currency_pair')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        if not all([currency_pair, start_date, end_date]):
            return jsonify({'success': False, 'error': '请提供所有必要参数'})
        
        # 导入优化模块
        from backend.backtest.backtest_optimum import StrategyOptimizer
        
        # 创建优化器实例
        optimizer = StrategyOptimizer()
        
        # 执行优化
        results = optimizer.optimize(
            currency_pair=currency_pair,
            start_date=start_date,
            end_date=end_date
        )
        
        if results['success']:
            return jsonify({
                'success': True,
                'original': results['original'],
                'optimized': results['optimized'],
                'improvement': results['improvement']
            })
        else:
            return jsonify({'success': False, 'error': results['error']})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({'success': False, 'error': '请输入用户名和密码'})
        
        # 连接数据库
        db = DatabaseManager()
        
        # 查询用户
        db.cur.execute("""
            SELECT user_id, username, password 
            FROM users.account 
            WHERE username = %s
        """, (username,))
        
        user = db.cur.fetchone()
        
        if not user:
            return jsonify({'success': False, 'error': '用户名不存在'})
        
        # 验证密码
        if not check_password_hash(user[2], password):
            return jsonify({'success': False, 'error': '密码错误'})
        
        # 更新最后登录时间
        db.cur.execute("""
            UPDATE users.account 
            SET last_login = CURRENT_TIMESTAMP 
            WHERE user_id = %s
        """, (user[0],))
        db.conn.commit()
        
        return jsonify({
            'success': True,
            'user': {
                'username': user[1],
                'user_id': user[0]
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    finally:
        if 'db' in locals():
            db.close()

@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        email = data.get('email')
        
        if not all([username, password, email]):
            return jsonify({'success': False, 'error': '请提供所有必填字段'})
            
        if len(password) > 100:
            return jsonify({'success': False, 'error': '密码长度不能超过100个字符'})
            
        # 对密码进行哈希处理
        hashed_password = generate_password_hash(password)
        
        db = DatabaseManager()
        success, result = db.register_user(username, hashed_password, email)
        db.close()
        
        if success:
            return jsonify({'success': True, 'message': '注册成功'})
        else:
            return jsonify({'success': False, 'error': result})
            
    except Exception as e:
        logger.error(f"注册API错误: {str(e)}")
        return jsonify({'success': False, 'error': '注册失败，请稍后重试'})

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

@app.route('/api/backtest', methods=['POST'])
def backtest():
    try:
        data = request.get_json()
        currency_pair = data.get('currency_pair')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        if not all([currency_pair, start_date, end_date]):
            return jsonify({'success': False, 'error': '请提供所有必要参数'})
        
        # 创建回测器实例
        backtester = ForexBacktester()
        
        # 执行回测
        results = backtester.run_backtest(
            currency_pair=currency_pair,
            start_date=start_date,
            end_date=end_date
        )
        
        if results:
            return jsonify({
                'success': True,
                'results': results
            })
        else:
            return jsonify({'success': False, 'error': '回测执行失败'})
        
    except Exception as e:
        logger.error(f"回测执行错误: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

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


@app.route('/api/signal_explanation', methods=['POST'])
def signal_explanation():
    try:
        data = request.get_json()
        currency_pair = data.get('currency_pair')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        if not all([currency_pair, start_date, end_date]):
            return jsonify({'success': False, 'error': '请提供所有必要参数'})
        
        # 导入信号解释模块
        from backend.explainer.request import get_signal_explanation_with_period
        
        # 获取信号解释
        result = get_signal_explanation_with_period(
            currency_pair=currency_pair,
            start_date=start_date,
            end_date=end_date
        )
        
        if result.get('success'):
            return jsonify({
                'success': True,
                'explanation': result['explanation']
            })
        else:
            return jsonify({'success': False, 'error': result.get('error', '生成信号解释失败')})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/risk_analysis', methods=['GET'])
def risk_analysis():
    try:
        # 读取风险分析数据
        risk_file = Path(__file__).parent / 'backend' / 'mulsignals' / 'currency_pair_risks.csv'
        
        # 如果文件不存在，返回错误
        if not risk_file.exists():
            return jsonify({
                'success': False, 
                'error': '风险数据文件不存在'
            })
        
        # 读取CSV文件
        risk_data = pd.read_csv(risk_file)
        
        # 转换为JSON格式
        risk_json = risk_data.to_dict(orient='records')
        
        return jsonify({
            'success': True,
            'data': risk_json
        })
        
    except Exception as e:
        logger.error(f"风险分析API错误: {str(e)}")
        return jsonify({
            'success': False, 
            'error': f'处理风险数据时出错: {str(e)}'
        })

@app.route('/static/data/<path:filename>')
def static_data(filename):
    # 安全检查，确保只能访问特定文件
    if filename == 'currency_pair_risks.csv':
        risk_file = Path(__file__).parent / 'backend' / 'mulsignals' / filename
        if risk_file.exists():
            return open(risk_file, 'r').read(), 200, {'Content-Type': 'text/csv'}
    return "文件不存在", 404

# 保留旧的路由作为兼容
@app.route('/api/single_backtest', methods=['POST'])
def single_backtest():
    return backtest()  # 调用新的回测函数

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
