# 导入 mysql.connector 模块以及 pooling 子模块，用于数据库连接池管理
import mysql.connector
from mysql.connector import pooling
# 引入 FastAPI 的 HTTPException 模块，用于抛出 HTTP 错误
from fastapi import HTTPException

# 配置数据库连接信息，包括主机、用户名、密码、数据库名及连接池大小
DATABASE_CONFIG = {
    "host": "localhost",  # 数据库主机地址
    "user": "root",       # 数据库用户名
    "password": "5201314", # 数据库密码
    "database": "chemical_server", # 数据库名称
    "pool_size": 5        # 连接池中最大连接数
}

# 初始化数据库连接池，通过 MySQLConnectionPool 类创建连接池实例
db_pool = pooling.MySQLConnectionPool(
    pool_name="auth_pool",  # 连接池的名称
    **DATABASE_CONFIG       # 使用上面定义的数据库连接配置
)

def get_db_conn():
    """
    获取数据库连接。
    如果连接池不可用，则抛出数据库连接失败异常。
    """
    try:
        # 从连接池获取数据库连接
        return db_pool.get_connection()
    except mysql.connector.Error as err:
        # 如果连接池获取连接失败，抛出 HTTP 500 错误，附带错误信息
        raise HTTPException(status_code=500, detail=f"数据库连接池获取连接失败: {err}")

def execute_query(query: str, params: tuple = ()):
    """
    执行查询操作，返回查询结果。
    """
    conn = None
    cursor = None
    try:
        # 获取数据库连接
        conn = get_db_conn()
        # 创建一个字典形式的游标，用于执行 SQL 查询
        cursor = conn.cursor(dictionary=True)
        # 执行查询操作
        cursor.execute(query, params)
        # 返回查询结果
        return cursor.fetchall()
    except mysql.connector.Error as err:
        # 如果查询过程中发生错误，抛出 HTTP 500 错误，附带错误信息
        raise HTTPException(status_code=500, detail=f"数据库查询失败: {err}")
    finally:
        # 确保游标和连接最终会被关闭
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def execute_update(query: str, params: tuple = ()):
    """
    执行更新操作（如插入、更新、删除）。
    Args:
        query (str): SQL更新语句
        params (tuple): 更新参数

    Returns:
        int: 对于INSERT操作返回lastrowid，对于其他操作返回受影响的行数
    """
    conn = None
    cursor = None
    try:
        # 获取数据库连接
        conn = get_db_conn()
        # 创建一个普通的游标，用于执行 SQL 更新操作
        cursor = conn.cursor()
        # 执行更新操作
        cursor.execute(query, params)
        # 提交更改到数据库
        conn.commit()

        # 根据操作类型返回不同的结果
        if query.strip().upper().startswith('INSERT'):
            # 对于INSERT操作，返回自增ID
            return cursor.lastrowid
        else:
            # 对于其他操作，返回受影响的行数
            return cursor.rowcount

    except mysql.connector.Error as err:
        # 如果更新操作失败，回滚事务
        if conn:
            conn.rollback()
        # 抛出 HTTP 500 错误，附带错误信息
        raise HTTPException(status_code=500, detail=f"数据库更新失败: {err}")
    finally:
        # 确保游标和连接最终会被关闭
        if cursor:
            cursor.close()
        if conn:
            conn.close()
