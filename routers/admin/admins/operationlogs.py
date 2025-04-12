# 引入 FastAPI 中的 APIRouter 和 HTTPException 模块，用于创建路由和处理异常
from fastapi import APIRouter, HTTPException, Depends, Query
# 引入 Pydantic 中的 BaseModel 类，用于定义请求体的数据结构和验证
from pydantic import BaseModel
# 从数据库模块导入 execute_query 和 execute_update 函数，用于执行查询和更新操作
from database import execute_query, execute_update
# 引入时间模块处理日期范围
from datetime import datetime, timedelta
# 引入 typing 模块中的 Optional 和 List 类型
from typing import Optional, List
# 引入管理员认证依赖函数
from routers.login import get_current_admin

# 初始化 APIRouter 实例，用于定义路由
router = APIRouter()

# 获取操作日志列表接口
@router.get("/admin/operation-logs", tags=["日志管理"])
async def get_operation_logs(
    page: int = Query(1, description="页码"),
    page_size: int = Query(10, description="每页数量"),
    admin_id: Optional[int] = Query(None, description="管理员ID筛选"),
    operation_type: Optional[str] = Query(None, description="操作类型筛选"),
    start_date: Optional[str] = Query(None, description="开始日期"),
    end_date: Optional[str] = Query(None, description="结束日期"),
    current_admin_id: Optional[int] = Query(None, description="当前管理员ID")
):
    """
    获取操作日志列表，支持分页和筛选
    """
    try:
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
        
        # 记录操作日志 (如果提供了管理员ID)
        if current_admin_id:
            try:
                execute_update(
                    """INSERT INTO operation_logs (admin_id, operation_type, operation_desc, created_at) 
                       VALUES (%s, %s, %s, NOW())""", 
                    (current_admin_id, "query", f"管理员{current_admin_id}查询操作日志")
                )
            except Exception as log_error:
                # 仅记录日志错误，不影响主流程
                print(f"记录操作日志失败: {str(log_error)}")
        
        return {
            "code": 200,
            "message": "获取操作日志成功",
            "data": {
                "logs": logs_list,
                "total": total_count
            }
        }
    except Exception as e:
        # 记录错误日志
        print(f"获取操作日志失败: {str(e)}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"获取操作日志失败: {str(e)}")

# 获取操作日志详情接口
@router.get("/admin/operation-logs/{log_id}", tags=["日志管理"])
async def get_operation_log_detail(
    log_id: int,
    current_admin_id: Optional[int] = Query(None, description="当前管理员ID")
):
    """
    获取操作日志详情
    """
    try:
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
        
        # 记录操作日志 (如果提供了管理员ID)
        if current_admin_id:
            try:
                execute_update(
                    """INSERT INTO operation_logs (admin_id, operation_type, operation_desc, created_at) 
                       VALUES (%s, %s, %s, NOW())""", 
                    (current_admin_id, "query", f"管理员{current_admin_id}查看日志{log_id}详情")
                )
            except Exception as log_error:
                # 仅记录日志错误，不影响主流程
                print(f"记录操作日志失败: {str(log_error)}")
        
        return {
            "code": 200,
            "message": "获取日志详情成功",
            "data": log
        }
    except Exception as e:
        # 记录错误日志
        print(f"获取日志详情失败: {str(e)}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"获取日志详情失败: {str(e)}")
