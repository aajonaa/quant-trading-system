from Database.Pgadmin4 import DatabaseManager
import psycopg2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_database():
    try:
        db = DatabaseManager()
        
        # 先删除现有的表和schema
        db.cur.execute("""
            DROP SCHEMA IF EXISTS users CASCADE;
        """)
        db.conn.commit()
        
        # 创建新的schema和表
        db.cur.execute("""
            CREATE SCHEMA users;
            
            CREATE TABLE users.account (
                user_id SERIAL PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                balance DECIMAL(10,2) DEFAULT 0.00,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                status VARCHAR(20) DEFAULT 'active'
            );
            
            CREATE TABLE users.transactions (
                transaction_id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users.account(user_id),
                amount DECIMAL(10,2) NOT NULL,
                type VARCHAR(20) NOT NULL,
                status VARCHAR(20) DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            );
        """)
        db.conn.commit()
        print("数据库初始化成功")
        
    except Exception as e:
        print(f"数据库初始化失败: {str(e)}")
        db.conn.rollback()
    finally:
        if db:
            db.close()

if __name__ == "__main__":
    init_database() 