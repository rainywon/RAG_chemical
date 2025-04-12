# 引入 FastAPI 中的 APIRouter 和 HTTPException 模块，用于创建路由和处理异常
from fastapi import APIRouter, HTTPException, Depends, Query
# 引入 Pydantic 中的 BaseModel 类，用于定义请求体的数据结构和验证
from pydantic import BaseModel
# 从数据库模块导入 execute_query 和 execute_update 函数，用于执行查询和更新操作
from database import execute_query
# 引入时间模块处理日期范围
from datetime import datetime, timedelta
# 引入 typing 模块中的 Optional 和 List 类型
from typing import Optional, List
# 引入管理员认证依赖函数
from routers.login import get_current_admin

# 初始化 APIRouter 实例，用于定义路由
router = APIRouter()

# 获取登录历史记录
@router.get("/admin/login-history/all", tags=["用户管理"])
async def get_login_history(
    page: int = Query(1, description="页码"),
    page_size: int = Query(10, description="每页数量"),
    user_id: Optional[int] = Query(None, description="用户ID筛选"),
    mobile: Optional[str] = Query(None, description="手机号筛选"),
    start_date: Optional[str] = Query(None, description="开始日期"),
    end_date: Optional[str] = Query(None, description="结束日期"),
    admin_id: Optional[int] = Query(None, description="管理员ID")
):
    """
    获取用户登录历史记录，支持分页和筛选
    """
    try:
        # 计算偏移量
        offset = (page - 1) * page_size
        
        # 构建基础查询SQL
        query = """
            SELECT 
                ut.id, ut.user_id, u.mobile, ut.created_at as login_time, 
                ut.is_valid, ut.expire_at, ut.device_info, ut.ip_address
            FROM 
                user_tokens ut
            JOIN 
                users u ON ut.user_id = u.user_id
            WHERE 1=1
        """
        params = []
        
        # 添加筛选条件
        if user_id:
            query += " AND ut.user_id = %s"
            params.append(user_id)
        
        if mobile:
            query += " AND u.mobile LIKE %s"
            params.append(f"%{mobile}%")
        
        if start_date:
            query += " AND DATE(ut.created_at) >= %s"
            params.append(start_date)
        
        if end_date:
            query += " AND DATE(ut.created_at) <= %s"
            params.append(end_date)
        
        # 查询符合条件的总记录数
        count_query = f"SELECT COUNT(*) as count FROM ({query}) as filtered_logins"
        count_result = execute_query(count_query, tuple(params))
        total_count = count_result[0]['count'] if count_result else 0
        
        # 添加排序和分页
        query += " ORDER BY ut.created_at DESC LIMIT %s OFFSET %s"
        params.append(page_size)
        params.append(offset)
        
        # 查询登录历史记录
        login_history = execute_query(query, tuple(params))
        
        # 处理日期时间格式
        for record in login_history:
            # 先判断状态，再格式化日期
            record['status'] = "有效" if record['is_valid'] == 1 and record['expire_at'] > datetime.now() else "已过期或无效"
            record['login_time'] = record['login_time'].strftime("%Y-%m-%d %H:%M:%S") if record['login_time'] else ""
            record['expire_at'] = record['expire_at'].strftime("%Y-%m-%d %H:%M:%S") if record['expire_at'] else ""
        
        # 如果提供了管理员ID，记录管理员操作
        if admin_id:
            try:
                execute_query(
                    """INSERT INTO operation_logs (admin_id, operation_type, operation_desc, created_at) 
                       VALUES (%s, %s, %s, NOW())""", 
                    (admin_id, "查询登录历史", f"管理员{admin_id}查询用户登录历史")
                )
            except Exception as log_error:
                # 记录日志错误，但不影响主流程
                print(f"记录操作日志失败: {str(log_error)}")
                
        return {
            "code": 200,
            "message": "获取登录历史记录成功",
            "data": {
                "logins": login_history,
                "total": total_count
            }
        }
    except Exception as e:
        # 记录错误日志
        print(f"获取登录历史记录失败: {str(e)}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"获取登录历史记录失败: {str(e)}")

# 获取指定用户的登录历史记录
@router.get("/admin/users/{user_id}/login-history", tags=["用户管理"])
async def get_user_login_history(
    user_id: int,
    page: int = Query(1, description="页码"),
    page_size: int = Query(10, description="每页数量"),
    admin_id: Optional[int] = Query(None, description="管理员ID")
):
    """
    获取指定用户的登录历史记录
    """
    try:
        # 验证用户是否存在
        user_check = execute_query(
            """SELECT * FROM users WHERE user_id = %s""", 
            (user_id,)
        )
        
        if not user_check:
            return {
                "code": 404,
                "message": "用户不存在"
            }
        
        # 计算偏移量
        offset = (page - 1) * page_size
        
        # 查询该用户的登录历史记录总数
        count_query = """
            SELECT COUNT(*) as count 
            FROM user_tokens 
            WHERE user_id = %s
        """
        count_result = execute_query(count_query, (user_id,))
        total_count = count_result[0]['count'] if count_result else 0
        
        # 查询该用户的登录历史记录
        login_query = """
            SELECT 
                id, user_id, created_at as login_time, 
                is_valid, expire_at, device_info, ip_address
            FROM 
                user_tokens
            WHERE 
                user_id = %s
            ORDER BY 
                created_at DESC
            LIMIT %s OFFSET %s
        """
        login_history = execute_query(login_query, (user_id, page_size, offset))
        
        # 处理日期时间格式
        for record in login_history:
            # 先判断状态，再格式化日期
            record['status'] = "有效" if record['is_valid'] == 1 and record['expire_at'] > datetime.now() else "已过期或无效"
            record['login_time'] = record['login_time'].strftime("%Y-%m-%d %H:%M:%S") if record['login_time'] else ""
            record['expire_at'] = record['expire_at'].strftime("%Y-%m-%d %H:%M:%S") if record['expire_at'] else ""
        
        # 如果提供了管理员ID，记录管理员操作
        if admin_id:
            try:
                execute_query(
                    """INSERT INTO operation_logs (admin_id, operation_type, operation_desc, created_at) 
                       VALUES (%s, %s, %s, NOW())""", 
                    (admin_id, "查询用户登录历史", f"管理员{admin_id}查询用户{user_id}的登录历史")
                )
            except Exception as log_error:
                # 记录日志错误，但不影响主流程
                print(f"记录操作日志失败: {str(log_error)}")
                
        return {
            "code": 200,
            "message": "获取用户登录历史记录成功",
            "data": {
                "logins": login_history,
                "total": total_count
            }
        }
    except Exception as e:
        # 记录错误日志
        print(f"获取用户登录历史记录失败: {str(e)}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"获取用户登录历史记录失败: {str(e)}")
