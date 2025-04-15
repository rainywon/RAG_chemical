# 引入 FastAPI 中的 APIRouter 和 HTTPException 模块，用于创建路由和处理异常
from fastapi import APIRouter, HTTPException, Depends, Query, Request
# 引入 Pydantic 中的 BaseModel 类，用于定义请求体的数据结构和验证
from pydantic import BaseModel
# 从数据库模块导入 execute_query 和 execute_update 函数，用于执行查询和更新操作
from database import execute_query, execute_update
# 引入时间模块处理日期范围
from datetime import datetime, timedelta
# 引入 typing 模块中的 Optional 和 List 类型
from typing import Optional, List
# 引入日志模块
import logging
import traceback
# 引入管理员认证依赖函数
from routers.user.login import get_current_admin
# 引入安全认证相关
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# 初始化 APIRouter 实例，用于定义路由
router = APIRouter()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化安全认证
security = HTTPBearer()

# 定义请求体的模型，用于改变用户状态
class UserStatusRequest(BaseModel):
    user_id: int
    status: int  # 0-禁用, 1-正常

# 安全记录管理员操作
def log_admin_operation(admin_id: int, operation_type: str, description: str):
    if not admin_id:
        return
    
    try:
        # 获取管理员姓名
        admin_result = execute_query(
            """SELECT full_name FROM admins WHERE admin_id = %s""",
            (admin_id,)
        )
        
        admin_name = admin_result[0]['full_name'] if admin_result else f"管理员{admin_id}"
        
        # 在描述前添加管理员姓名
        full_description = f"{admin_name}{description}"
        
        execute_update(
            """INSERT INTO operation_logs (admin_id, operation_type, operation_desc, created_at) 
               VALUES (%s, %s, %s, NOW())""",
            (admin_id, operation_type, full_description)
        )
    except Exception as e:
        # 记录错误但不中断主要流程
        logger.error(f"记录操作日志失败: {str(e)}")

# 获取用户列表接口
@router.get("/admin/users", tags=["用户管理"])
async def get_users(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    page: int = Query(1, description="页码"),
    page_size: int = Query(10, description="每页数量"),
    mobile: Optional[str] = Query(None, description="手机号筛选"),
    status: Optional[str] = Query(None, description="状态筛选"),
    start_date: Optional[str] = Query(None, description="注册开始日期"),
    end_date: Optional[str] = Query(None, description="注册结束日期")
):
    """
    获取用户列表，支持分页和筛选
    """
    try:
        # 获取管理员ID
        admin_id = await get_current_admin(credentials)
        
        # 计算偏移量
        offset = (page - 1) * page_size
        
        # 构建基础查询SQL
        query = """
            SELECT 
                user_id, mobile, register_time, last_login_time, status, theme_preference
            FROM 
                users
            WHERE 1=1
        """
        params = []
        
        # 添加筛选条件
        if mobile:
            query += " AND mobile LIKE %s"
            params.append(f"%{mobile}%")
        
        if status:
            query += " AND status = %s"
            params.append(int(status))
        
        if start_date:
            query += " AND DATE(register_time) >= %s"
            params.append(start_date)
        
        if end_date:
            query += " AND DATE(register_time) <= %s"
            params.append(end_date)
        
        # 查询符合条件的总记录数
        count_query = f"SELECT COUNT(*) as count FROM ({query}) as filtered_users"
        count_result = execute_query(count_query, tuple(params))
        total_count = count_result[0]['count'] if count_result else 0
        
        # 添加排序和分页
        query += " ORDER BY register_time DESC LIMIT %s OFFSET %s"
        params.append(page_size)
        params.append(offset)
        
        # 查询用户列表
        user_list = execute_query(query, tuple(params))
        
        # 处理日期时间格式
        for user in user_list:
            user['register_time'] = user['register_time'].strftime("%Y-%m-%d %H:%M:%S") if user['register_time'] else ""
            user['last_login_time'] = user['last_login_time'].strftime("%Y-%m-%d %H:%M:%S") if user['last_login_time'] else ""
        
        # 记录操作日志
        log_admin_operation(admin_id, "查询", "查询用户列表")
        
        return {
            "code": 200,
            "message": "获取用户列表成功",
            "data": {
                "users": user_list,
                "total": total_count
            }
        }
    except Exception as e:
        # 记录错误日志
        logger.error(f"获取用户列表失败: {str(e)}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"获取用户列表失败: {str(e)}")

# 更改用户状态接口
@router.post("/admin/users/status", tags=["用户管理"])
async def update_user_status(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    user_status: UserStatusRequest = None
):
    """
    更新用户账户状态
    """
    try:
        # 获取管理员ID
        admin_id = await get_current_admin(credentials)
        
        # 验证用户是否存在
        user_check = execute_query(
            """SELECT * FROM users WHERE user_id = %s""", 
            (user_status.user_id,)
        )
        
        if not user_check:
            return {
                "code": 404,
                "message": "用户不存在"
            }
        
        # 更新用户状态
        execute_update(
            """UPDATE users SET status = %s WHERE user_id = %s""", 
            (user_status.status, user_status.user_id)
        )
        
        # 记录操作日志
        operation_type = "启用用户" if user_status.status == 1 else "禁用用户"
        log_admin_operation(admin_id, "更新", f"{operation_type}{user_status.user_id}")
        
        return {
            "code": 200,
            "message": "更新用户状态成功"
        }
    except Exception as e:
        # 记录错误日志
        logger.error(f"更新用户状态失败: {str(e)}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"更新用户状态失败: {str(e)}")

# 获取用户详情接口
@router.get("/admin/users/{user_id}", tags=["用户管理"])
async def get_user_detail(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    user_id: int = None
):
    """
    获取用户详细信息
    """
    try:
        # 获取管理员ID
        admin_id = await get_current_admin(credentials)
        
        # 查询用户详情
        user_detail = execute_query(
            """SELECT 
                   user_id, mobile, register_time, last_login_time, status, theme_preference
               FROM 
                   users
               WHERE 
                   user_id = %s""", 
            (user_id,)
        )
        
        if not user_detail:
            return {
                "code": 404,
                "message": "用户不存在"
            }
        
        # 处理日期时间格式
        user = user_detail[0]
        user['register_time'] = user['register_time'].strftime("%Y-%m-%d %H:%M:%S") if user['register_time'] else ""
        user['last_login_time'] = user['last_login_time'].strftime("%Y-%m-%d %H:%M:%S") if user['last_login_time'] else ""
        
        # 记录操作日志
        log_admin_operation(admin_id, "查询", f"查看用户{user_id}详情")
        
        return {
            "code": 200,
            "message": "获取用户详情成功",
            "data": user
        }
    except Exception as e:
        # 记录错误日志
        logger.error(f"获取用户详情失败: {str(e)}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"获取用户详情失败: {str(e)}") 