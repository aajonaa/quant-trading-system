from flask import Flask, render_template, request, jsonify, session
from backend.backtest.backtest_optimum import BacktestOptimizer
from pathlib import Path
import logging
import sys
import pandas as pd
from werkzeug.security import generate_password_hash

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
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        email = data.get('email')
        
        if not all([username, password, email]):
            return jsonify({'error': '请提供所有必填字段'})
        
        # 连接数据库
        db = DatabaseManager()
        
        try:
            # 检查用户名是否已存在
            db.cur.execute("SELECT username FROM users.account WHERE username = %s", (username,))
            if db.cur.fetchone():
                return jsonify({'error': '用户名已存在'})
            
            # 检查邮箱是否已存在
            db.cur.execute("SELECT email FROM users.account WHERE email = %s", (email,))
            if db.cur.fetchone():
                return jsonify({'error': '邮箱已被注册'})
            
            # 加密密码
            hashed_password = generate_password_hash(password)
            
            # 插入新用户
            db.cur.execute("""
                INSERT INTO users.account (username, password, email)
                VALUES (%s, %s, %s)
                RETURNING user_id
            """, (username, hashed_password, email))
            
            user_id = db.cur.fetchone()[0]
            db.conn.commit()
            
            return jsonify({
                'success': True,
                'message': '注册成功',
                'user_id': user_id
            })
            
        finally:
            db.close()
            
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
        backtester = ForexBacktester()
        results = backtester.run_backtest(
            currency_pair=data['currency_pair'],
            start_date=data['start_date'],
            end_date=data['end_date']
        )
        return jsonify({
            'success': True,
            'results': results
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

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
