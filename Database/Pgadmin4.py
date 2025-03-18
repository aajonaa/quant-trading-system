import psycopg2
import os
import pandas as pd
from pathlib import Path

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
        # 创建所需的schema
        schemas = [
            'feature_engineering',  # 特征工程
            'forex_data',          # 外汇数据
            'macro_data',          # 宏观数据
            'backtest_signals',    # 回测信号
            'multi_risk'           # 多货币风险
        ]
        
        for schema in schemas:
            self.cur.execute(f"""
                CREATE SCHEMA IF NOT EXISTS {schema};
            """)
        self.conn.commit()

    def create_tables(self):
        # 在每个schema下创建对应的表
        # 特征工程表
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS feature_engineering.features (
                date DATE,
                currency_pair VARCHAR(10),
                feature_name VARCHAR(50),
                value FLOAT,
                PRIMARY KEY (date, currency_pair, feature_name)
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
                indicator_name VARCHAR(50),
                value FLOAT,
                country VARCHAR(50),
                PRIMARY KEY (date, indicator_name, country)
            );
        """)

        # 回测信号表
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS backtest_signals.signals (
                date DATE,
                currency_pair VARCHAR(10),
                signal FLOAT,
                PRIMARY KEY (date, currency_pair)
            );
        """)

        # 多货币风险表
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS multi_risk.analysis (
                date DATE,
                pair_combination VARCHAR(20),
                correlation FLOAT,
                risk_score FLOAT,
                volatility FLOAT,
                PRIMARY KEY (date, pair_combination)
            );
        """)

        self.conn.commit()

    def import_data(self, project_root):
        # 导入特征工程数据
        fe_path = Path(project_root) / 'backend' / 'FE'
        for csv_file in fe_path.glob('*.csv'):
            df = pd.read_csv(csv_file)
            # 根据实际CSV文件结构调整以下代码
            for _, row in df.iterrows():
                self.cur.execute("""
                    INSERT INTO feature_engineering.features 
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT DO NOTHING;
                """, (row['date'], row['currency_pair'], row['feature_name'], row['value']))

        # 导入外汇数据
        forex_path = Path(project_root) / 'try' / 'data'
        for csv_file in forex_path.glob('*.csv'):
            df = pd.read_csv(csv_file)
            for _, row in df.iterrows():
                self.cur.execute("""
                    INSERT INTO forex_data.prices 
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING;
                """, (row['date'], csv_file.stem, row['open'], row['high'], 
                     row['low'], row['close'], row['volume']))

        # 导入宏观数据
        macro_path = Path(project_root) / 'try' / 'macro_data'
        for csv_file in macro_path.glob('*.csv'):
            df = pd.read_csv(csv_file)
            for _, row in df.iterrows():
                self.cur.execute("""
                    INSERT INTO macro_data.indicators 
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT DO NOTHING;
                """, (row['date'], row['indicator'], row['value'], row['country']))

        # 导入回测信号
        signals_path = Path(project_root) / 'backend' / 'signals'
        for csv_file in signals_path.glob('*.csv'):
            df = pd.read_csv(csv_file)
            for _, row in df.iterrows():
                self.cur.execute("""
                    INSERT INTO backtest_signals.signals 
                    VALUES (%s, %s, %s)
                    ON CONFLICT DO NOTHING;
                """, (row['date'], csv_file.stem, row['signal']))

        # 导入多货币风险数据
        multi_risk_path = Path(project_root) / 'backend' / 'multisignals'
        for csv_file in multi_risk_path.glob('*.csv'):
            df = pd.read_csv(csv_file)
            for _, row in df.iterrows():
                self.cur.execute("""
                    INSERT INTO multi_risk.analysis 
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING;
                """, (row['date'], row['pair_combination'], row['correlation'],
                     row['risk_score'], row['volatility']))

        # 提交所有更改
        self.conn.commit()

    def close(self):
        self.cur.close()
        self.conn.close()

if __name__ == "__main__":
    # 使用示例
    db_manager = DatabaseManager()
    
    try:
        # 创建schema
        db_manager.create_schema()
        
        # 创建表
        db_manager.create_tables()
        
        # 导入数据
        project_root = "D:/PycharmProjects/FlaskProject替身"  # 替换为你的项目根目录
        db_manager.import_data(project_root)
        
        print("数据库初始化和数据导入完成！")
    
    except Exception as e:
        print(f"发生错误: {str(e)}")
    
    finally:
        db_manager.close()