# import psycopg2
#
#
# # Set up the database connection parameters
# conn = psycopg2.connect(
#     dbname="postgres",
#     user='postgres',
#     password='123456',
#     host='localhost',
#     port='5432'
# )
#
#
# # Create a cursor object to interact with the database
# cur = conn.cursor()
#
#
# # Execute a cursor object to interact with the database
# cur.execute("SELECT current_user;")
# rows = cur.fetchall()
#
#
# # Print the result
# for row in rows:
#     print(row)
#
#
# # Close the cursor and connection after the query
# cur.close()
# conn.close()

import psycopg2
import csv

# 设置数据库连接
conn = psycopg2.connect(
    dbname="quant_trading_db",
    user='postgres',
    password='jonawong.',
    host='localhost',
    port='5432'
)
cur = conn.cursor()

# 1. 清空表中原有数据（TRUNCATE 比 DELETE 更快且不记录日志）
cur.execute("TRUNCATE TABLE cnyeur;")

# 2. 从 CSV 导入新数据
csv_path = 'D:/VSCode/Matlab_env/PostgresSQL/NewFiles/CNYEUR.csv'

# 方法一：使用 COPY 命令（推荐）
with open(csv_path, 'r') as f:
    next(f)  # 跳过 CSV 的标题行（如果存在）
    cur.copy_from(f, 'cnyeur', sep=',', null='')

# 方法二：逐行插入（如需数据清洗可使用此方法）
# with open(csv_path, 'r') as f:
#     reader = csv.reader(f)
#     next(reader)  # 跳过标题行
#     for row in reader:
#         cur.execute("INSERT INTO cnyeur VALUES (%s, %s)", row)

# 提交事务并关闭连接
conn.commit()
cur.close()
conn.close()

print("数据已完全更新！")