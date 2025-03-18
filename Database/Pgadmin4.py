import psycopg2
import os
import pandas as pd
from pathlib import Path
from datetime import datetime

class DatabaseManager:
    def __init__(self):
        self.conn = psycopg2.connect(
            dbname="postgres",
            user='postgres',
            password='123456',
            host='localhost',
            port='5432'
        )
        self.cur = self.conn.cursor()

    def create_schema(self):
        """创建所需的schema"""
        schemas = [
            'feature_engineering',  # 特征工程
            'forex_data',          # 外汇数据
            'macro_data',          # 宏观数据
            'backtest_signals',    # 回测信号
            'multi_risk',          # 多货币风险
            'users'               # 添加用户schema
        ]
        
        for schema in schemas:
            self.cur.execute(f"""
                DROP SCHEMA IF EXISTS {schema} CASCADE;
                CREATE SCHEMA {schema};
            """)
        self.conn.commit()
        print("所有Schema已重新创建")

    def create_tables(self):
        """创建所需的表"""
        # 先创建schema
        self.create_schema()
        
        # 创建用户表
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS users.account (
                user_id SERIAL PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                password VARCHAR(100) NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                balance DECIMAL(10,2) DEFAULT 0.00,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                status VARCHAR(20) DEFAULT 'active'
            );
            
            CREATE TABLE IF NOT EXISTS users.transactions (
                transaction_id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users.account(user_id),
                amount DECIMAL(10,2) NOT NULL,
                type VARCHAR(20) NOT NULL,
                status VARCHAR(20) DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            );
        """)
        
        # 外汇数据表
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS forex_data.prices (
                date DATE,
                currency_pair VARCHAR(10),
                open FLOAT,
                high FLOAT,
                low FLOAT,
                close FLOAT,
                PRIMARY KEY (date, currency_pair)
            );
        """)

        # 宏观数据表
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS macro_data.indicators (
                date DATE,
                country VARCHAR(10),
                indicator VARCHAR(50),
                value FLOAT,
                PRIMARY KEY (date, country, indicator)
            );
        """)

        # 回测信号表
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS backtest_signals.signals (
                date DATE,
                currency_pair VARCHAR(10),
                price FLOAT,
                signal FLOAT,
                PRIMARY KEY (date, currency_pair)
            );
        """)

        # 多货币风险表
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS multi_risk.analysis (
                pair_combination VARCHAR(20),
                correlation FLOAT,
                volatility FLOAT,
                signal_consistency FLOAT,
                risk_score FLOAT,
                risk_level VARCHAR(20),
                trading_suggestion TEXT,
                PRIMARY KEY (pair_combination)
            );
        """)

        self.conn.commit()
        print("所有表格已创建")

    def format_date(self, date_str):
        """转换日期字符串为PostgreSQL支持的格式"""
        try:
            return pd.to_datetime(str(date_str)).strftime('%Y-%m-%d')
        except Exception as e:
            print(f"日期转换错误 {date_str}: {str(e)}")
            return None

    def import_data(self, backend_path, try_path):
        """导入数据"""
        try:
            # 获取脚本所在目录的绝对路径
            script_dir = Path(__file__).parent.parent
            backend_path = script_dir / backend_path
            try_path = script_dir / try_path

            print(f"正在从以下路径导入数据：")
            print(f"Backend路径: {backend_path}")
            print(f"Try路径: {try_path}")

            # 导入外汇数据
            print("\n正在导入外汇数据...")
            forex_path = try_path / 'data'
            if forex_path.exists():
                print(f"找到外汇数据目录: {forex_path}")
                for csv_file in forex_path.glob('*.csv'):
                    try:
                        df = pd.read_csv(csv_file)
                        currency_pair = csv_file.stem
                        print(f"正在处理 {currency_pair} 的数据...")
                        
                        for index, row in df.iterrows():
                            date = self.format_date(row['Date'])
                            if date:
                                self.cur.execute("""
                                    INSERT INTO forex_data.prices 
                                    VALUES (%s, %s, %s, %s, %s, %s)
                                    ON CONFLICT DO NOTHING;
                                """, (date, currency_pair, 
                                     float(row['Open']), float(row['High']), 
                                     float(row['Low']), float(row['Close'])))
                        print(f"成功导入 {len(df)} 条 {currency_pair} 数据")
                    except Exception as e:
                        print(f"处理文件 {csv_file} 时出错: {str(e)}")
            else:
                print(f"警告: 找不到外汇数据目录 {forex_path}")

            # 导入宏观数据
            print("\n正在导入宏观数据...")
            macro_path = try_path / 'macro_data'
            if macro_path.exists():
                print(f"找到宏观数据目录: {macro_path}")
                for csv_file in macro_path.glob('*.csv'):
                    try:
                        df = pd.read_csv(csv_file)
                        indicator = csv_file.stem
                        country = indicator.split('_')[0]
                        print(f"正在处理 {indicator} 的数据...")
                        
                        success_count = 0
                        for index, row in df.iterrows():
                            if pd.notna(row['date']) and pd.notna(row.iloc[1]):
                                date = self.format_date(row['date'])
                                if date:
                                    self.cur.execute("""
                                        INSERT INTO macro_data.indicators 
                                        VALUES (%s, %s, %s, %s)
                                        ON CONFLICT DO NOTHING;
                                    """, (date, country, indicator, float(row.iloc[1])))
                                    success_count += 1
                        print(f"成功导入 {success_count} 条 {indicator} 数据")
                    except Exception as e:
                        print(f"处理文件 {csv_file} 时出错: {str(e)}")
            else:
                print(f"警告: 找不到宏观数据目录 {macro_path}")

            # 导入回测信号
            print("\n正在导入回测信号...")
            signals_path = backend_path / 'signals'
            if signals_path.exists():
                print(f"找到信号数据目录: {signals_path}")
                for csv_file in signals_path.glob('*.csv'):
                    try:
                        df = pd.read_csv(csv_file)
                        currency_pair = csv_file.stem.split('_')[0]  # 从文件名获取货币对
                        print(f"正在处理 {currency_pair} 的信号...")
                        
                        success_count = 0
                        for index, row in df.iterrows():
                            date = self.format_date(row['Date'])  # 注意这里使用'Date'而不是'date'
                            if date:
                                self.cur.execute("""
                                    INSERT INTO backtest_signals.signals 
                                    VALUES (%s, %s, %s, %s)
                                    ON CONFLICT DO NOTHING;
                                """, (date, currency_pair, float(row['Price']), float(row['Signal'])))
                                success_count += 1
                        print(f"成功导入 {success_count} 条 {currency_pair} 信号")
                    except Exception as e:
                        print(f"处理文件 {csv_file} 时出错: {str(e)}")
            else:
                print(f"警告: 找不到信号数据目录 {signals_path}")

            # 导入多货币风险数据
            print("\n正在导入多货币风险数据...")
            multi_risk_path = backend_path / 'mulsignals'
            if multi_risk_path.exists():
                print(f"找到风险数据目录: {multi_risk_path}")
                for csv_file in multi_risk_path.glob('*.csv'):
                    try:
                        df = pd.read_csv(csv_file)
                        print(f"正在处理 {csv_file.stem} 的风险数据...")
                        
                        success_count = 0
                        for index, row in df.iterrows():
                            self.cur.execute("""
                                INSERT INTO multi_risk.analysis 
                                VALUES (%s, %s, %s, %s, %s, %s, %s)
                                ON CONFLICT DO NOTHING;
                            """, (row['货币对组合'], float(row['相关系数']), 
                                 float(row['组合波动率']), float(row['信号一致性']),
                                 float(row['风险得分']), row['风险等级'],
                                 row['交易建议']))
                            success_count += 1
                        print(f"成功导入 {success_count} 条风险数据")
                    except Exception as e:
                        print(f"处理文件 {csv_file} 时出错: {str(e)}")
            else:
                print(f"警告: 找不到风险数据目录 {multi_risk_path}")

            self.conn.commit()
            print("\n所有数据导入完成！")

        except Exception as e:
            print(f"导入过程中发生错误: {str(e)}")
            self.conn.rollback()

    def close(self):
        self.cur.close()
        self.conn.close()

if __name__ == "__main__":
    db_manager = DatabaseManager()
    
    try:
        # 创建schema和表
        db_manager.create_schema()
        db_manager.create_tables()
        
        # 导入数据
        backend_dir = "backend"
        try_dir = "try"
        db_manager.import_data(backend_dir, try_dir)
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
    
    finally:
        db_manager.close()
        print("\n数据库连接已关闭")