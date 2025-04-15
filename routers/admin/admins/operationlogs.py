# 引入 FastAPI 中的 APIRouter 和 HTTPException 模块，用于创建路由和处理异常
from fastapi import APIRouter, HTTPException, Query, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
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

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化 APIRouter 实例，用于定义路由
router = APIRouter()
security = HTTPBearer()

# 获取当前管理员ID的函数
async def get_current_admin(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        # 从token中获取管理员ID
        token = credentials.credentials
        result = execute_query(
            "SELECT admin_id FROM admin_tokens WHERE token = %s AND is_valid = 1 AND expire_at > NOW()",
            (token,)
        )
        
        if not result:
            raise HTTPException(status_code=401, detail="无效的token或token已过期")
            
        return result[0]['admin_id']
    except Exception as e:
        logger.error(f"获取管理员ID失败: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="获取管理员ID失败")

# 记录管理员操作的函数
async def log_admin_operation(admin_id: int, operation_type: str, operation_desc: str):
    try:
        execute_update(
            """INSERT INTO operation_logs (admin_id, operation_type, operation_desc, created_at) 
               VALUES (%s, %s, %s, NOW())""", 
            (admin_id, operation_type, operation_desc)
        )
    except Exception as e:
        logger.error(f"记录操作日志失败: {str(e)}\n{traceback.format_exc()}")

# 获取操作日志列表接口
@router.get("/admin/operation-logs", tags=["日志管理"])
async def get_operation_logs(
    page: int = Query(1, description="页码"),
    page_size: int = Query(10, description="每页数量"),
    admin_id: Optional[int] = Query(None, description="管理员ID筛选"),
    operation_type: Optional[str] = Query(None, description="操作类型筛选"),
    start_date: Optional[str] = Query(None, description="开始日期"),
    end_date: Optional[str] = Query(None, description="结束日期"),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    获取操作日志列表，支持分页和筛选
    """
    try:
        # 获取当前管理员ID
        current_admin_id = await get_current_admin(credentials)
        
        # 计算偏移量
        offset = (page - 1) * page_size
        
        # 构建基础查询SQL
        query = """
            SELECT 
                log_id, admin_id, operation_type, operation_desc,
                ip_address, user_agent, created_at
            FROM 
                operation_logs
            WHERE 1=1
        """
        params = []
        
        # 添加筛选条件
        if admin_id:
            query += " AND admin_id = %s"
            params.append(admin_id)
        
        if operation_type:
            query += " AND operation_type = %s"
            params.append(operation_type)
        
        if start_date:
            query += " AND DATE(created_at) >= %s"
            params.append(start_date)
        
        if end_date:
            query += " AND DATE(created_at) <= %s"
            params.append(end_date)
        
        # 查询符合条件的总记录数
        count_query = f"SELECT COUNT(*) as count FROM ({query}) as filtered_logs"
        count_result = execute_query(count_query, tuple(params))
        total_count = count_result[0]['count'] if count_result else 0
        
        # 添加排序和分页
        query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
        params.append(page_size)
        params.append(offset)
        
        # 查询操作日志列表
        logs_list = execute_query(query, tuple(params))
        
        # 处理日期时间格式
        for log in logs_list:
            log['created_at'] = log['created_at'].strftime("%Y-%m-%d %H:%M:%S") if log['created_at'] else None
        
        # 记录操作日志
        await log_admin_operation(current_admin_id, "查询", f"管理员{current_admin_id}查询操作日志")
        
        return {
            "code": 200,
            "message": "获取操作日志成功",
            "data": {
                "logs": logs_list,
                "total": total_count
            }
        }
    except Exception as e:
        logger.error(f"获取操作日志失败: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"获取操作日志失败: {str(e)}")

# 获取操作日志详情接口
@router.get("/admin/operation-logs/{log_id}", tags=["日志管理"])
async def get_operation_log_detail(
    log_id: int,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    获取操作日志详情
    """
    try:
        # 获取当前管理员ID
        current_admin_id = await get_current_admin(credentials)
        
        # 查询日志详情
        log_detail = execute_query(
            """SELECT 
                   log_id, admin_id, operation_type, operation_desc,
                   ip_address, user_agent, created_at
               FROM 
                   operation_logs
               WHERE 
                   log_id = %s""", 
            (log_id,)
        )
        
        if not log_detail:
            return {
                "code": 404,
                "message": "日志不存在"
            }
        
        # 处理日期时间格式
        log = log_detail[0]
        log['created_at'] = log['created_at'].strftime("%Y-%m-%d %H:%M:%S") if log['created_at'] else None
        
        # 记录操作日志
        await log_admin_operation(current_admin_id, "查询", f"管理员{current_admin_id}查看日志{log_id}详情")
        
        return {
            "code": 200,
            "message": "获取日志详情成功",
            "data": log
        }
    except Exception as e:
        logger.error(f"获取日志详情失败: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"获取日志详情失败: {str(e)}")
