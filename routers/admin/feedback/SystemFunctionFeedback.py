# 引入 FastAPI 中的 APIRouter 和 HTTPException 模块，用于创建路由和处理异常
from fastapi import APIRouter, HTTPException, Query
# 引入 Pydantic 中的 BaseModel 类，用于定义请求体的数据结构和验证
from pydantic import BaseModel
# 从数据库模块导入 execute_query 和 execute_update 函数，用于执行查询和更新操作
from database import execute_query, execute_update
# 引入时间模块处理日期范围
from datetime import datetime, timedelta
# 引入 typing 模块中的 Optional 和 List 类型
from typing import Optional, List

# 初始化 APIRouter 实例，用于定义路由
router = APIRouter()

# 定义请求体的模型，用于更新反馈处理状态
class UpdateFeedbackStatusRequest(BaseModel):
    status: str  # 反馈状态：pending, processing, resolved, rejected
    admin_reply: Optional[str] = None  # 管理员回复内容

# 获取系统功能反馈列表接口
@router.get("/admin/feedback/system", tags=["反馈管理"])
async def get_system_feedback_list(
    page: int = Query(1, description="页码"),
    page_size: int = Query(10, description="每页数量"),
    feedback_type: Optional[str] = Query(None, description="反馈类型筛选"),
    status: Optional[str] = Query(None, description="处理状态筛选"),
    start_date: Optional[str] = Query(None, description="开始日期"),
    end_date: Optional[str] = Query(None, description="结束日期"),
    keyword: Optional[str] = Query(None, description="关键词搜索"),
    current_admin_id: Optional[int] = Query(None, description="当前管理员ID")
):
    """
    获取系统功能反馈列表，支持分页和筛选
    从user_feedback表获取数据
    """
    try:
        # 计算偏移量
        offset = (page - 1) * page_size
        
        # 构建基础查询SQL
        query = """
            SELECT 
                feedback_id, user_id, feedback_type, feedback_content,
                created_at, status, admin_reply, replied_at
            FROM 
                user_feedback
            WHERE 1=1
        """
        params = []
        
        # 添加筛选条件
        if feedback_type:
            query += " AND feedback_type = %s"
            params.append(feedback_type)
        
        if status:
            query += " AND status = %s"
            params.append(status)
        
        if start_date:
            query += " AND DATE(created_at) >= %s"
            params.append(start_date)
        
        if end_date:
            query += " AND DATE(created_at) <= %s"
            params.append(end_date)
        
        if keyword:
            query += " AND feedback_content LIKE %s"
            params.append(f"%{keyword}%")
        
        # 查询符合条件的总记录数
        count_query = f"SELECT COUNT(*) as count FROM ({query}) as filtered_feedback"
        count_result = execute_query(count_query, tuple(params))
        total_count = count_result[0]['count'] if count_result else 0
        
        # 添加排序和分页
        query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
        params.append(page_size)
        params.append(offset)
        
        # 查询反馈列表
        feedback_list = execute_query(query, tuple(params))
        
        # 处理日期时间格式
        for feedback in feedback_list:
            feedback['created_at'] = feedback['created_at'].strftime("%Y-%m-%d %H:%M:%S") if feedback['created_at'] else None
            feedback['replied_at'] = feedback['replied_at'].strftime("%Y-%m-%d %H:%M:%S") if feedback['replied_at'] else None
        
        # 记录操作日志 (如果提供了管理员ID)
        if current_admin_id:
            try:
                execute_update(
                    """INSERT INTO operation_logs (admin_id, operation_type, operation_desc, created_at) 
                       VALUES (%s, %s, %s, NOW())""", 
                    (current_admin_id, "查询", f"管理员{current_admin_id}查询系统功能反馈列表")
                )
            except Exception as log_error:
                # 仅记录日志错误，不影响主流程
                print(f"记录操作日志失败: {str(log_error)}")
        
        return {
            "code": 200,
            "message": "获取系统功能反馈列表成功",
            "data": {
                "list": feedback_list,
                "total": total_count
            }
        }
    except Exception as e:
        # 记录错误日志
        print(f"获取系统功能反馈列表失败: {str(e)}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"获取系统功能反馈列表失败: {str(e)}")

# 获取系统功能反馈状态统计接口
@router.get("/admin/feedback/system/stats", tags=["反馈管理"])
async def get_system_feedback_stats(
    feedback_type: Optional[str] = Query(None, description="反馈类型筛选"),
    start_date: Optional[str] = Query(None, description="开始日期"),
    end_date: Optional[str] = Query(None, description="结束日期"),
    keyword: Optional[str] = Query(None, description="关键词搜索"),
    current_admin_id: Optional[int] = Query(None, description="当前管理员ID")
):
    """
    获取系统功能反馈状态统计
    """
    try:
        # 构建基础查询SQL
        query = """
            SELECT 
                status, COUNT(*) as count
            FROM 
                user_feedback
            WHERE 1=1
        """
        params = []
        
        # 添加筛选条件
        if feedback_type:
            query += " AND feedback_type = %s"
            params.append(feedback_type)
        
        if start_date:
            query += " AND DATE(created_at) >= %s"
            params.append(start_date)
        
        if end_date:
            query += " AND DATE(created_at) <= %s"
            params.append(end_date)
        
        if keyword:
            query += " AND feedback_content LIKE %s"
            params.append(f"%{keyword}%")
        
        # 按状态分组
        query += " GROUP BY status"
        
        # 执行查询
        status_counts = execute_query(query, tuple(params))
        
        # 处理结果为字典格式
        result = {}
        for item in status_counts:
            result[item['status']] = item['count']
        
        # 确保所有状态都有值，即使是0
        for status in ['pending', 'processing', 'resolved', 'rejected']:
            if status not in result:
                result[status] = 0
        
        # 记录操作日志 (如果提供了管理员ID)
        if current_admin_id:
            try:
                execute_update(
                    """INSERT INTO operation_logs (admin_id, operation_type, operation_desc, created_at) 
                       VALUES (%s, %s, %s, NOW())""", 
                    (current_admin_id, "查询", f"管理员{current_admin_id}查询系统功能反馈统计")
                )
            except Exception as log_error:
                # 仅记录日志错误，不影响主流程
                print(f"记录操作日志失败: {str(log_error)}")
        
        return {
            "code": 200,
            "message": "获取系统功能反馈状态统计成功",
            "data": {
                "status_counts": result
            }
        }
    except Exception as e:
        # 记录错误日志
        print(f"获取系统功能反馈状态统计失败: {str(e)}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"获取系统功能反馈状态统计失败: {str(e)}")

# 获取系统功能反馈详情接口
@router.get("/admin/feedback/system/{feedback_id}", tags=["反馈管理"])
async def get_system_feedback_detail(
    feedback_id: int,
    current_admin_id: Optional[int] = Query(None, description="当前管理员ID")
):
    """
    获取系统功能反馈详情
    """
    try:
        # 查询反馈详情
        feedback_detail = execute_query(
            """SELECT 
                   feedback_id, user_id, feedback_type, feedback_content,
                   created_at, status, admin_reply, replied_at
               FROM 
                   user_feedback
               WHERE 
                   feedback_id = %s""", 
            (feedback_id,)
        )
        
        if not feedback_detail:
            return {
                "code": 404,
                "message": "反馈不存在"
            }
        
        # 处理日期时间格式
        feedback = feedback_detail[0]
        feedback['created_at'] = feedback['created_at'].strftime("%Y-%m-%d %H:%M:%S") if feedback['created_at'] else None
        feedback['replied_at'] = feedback['replied_at'].strftime("%Y-%m-%d %H:%M:%S") if feedback['replied_at'] else None
        
        # 记录操作日志 (如果提供了管理员ID)
        if current_admin_id:
            try:
                execute_update(
                    """INSERT INTO operation_logs (admin_id, operation_type, operation_desc, created_at) 
                       VALUES (%s, %s, %s, NOW())""", 
                    (current_admin_id, "查询", f"管理员{current_admin_id}查看系统功能反馈{feedback_id}详情")
                )
            except Exception as log_error:
                # 仅记录日志错误，不影响主流程
                print(f"记录操作日志失败: {str(log_error)}")
        
        return {
            "code": 200,
            "message": "获取反馈详情成功",
            "data": feedback
        }
    except Exception as e:
        # 记录错误日志
        print(f"获取反馈详情失败: {str(e)}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"获取反馈详情失败: {str(e)}")

# 更新系统功能反馈状态接口
@router.put("/admin/feedback/system/{feedback_id}", tags=["反馈管理"])
async def update_system_feedback_status(
    feedback_id: int,
    request: UpdateFeedbackStatusRequest,
    current_admin_id: Optional[int] = Query(None, description="当前管理员ID")
):
    """
    更新系统功能反馈处理状态和回复
    """
    try:
        # 验证反馈是否存在
        feedback_exists = execute_query(
            "SELECT feedback_id FROM user_feedback WHERE feedback_id = %s",
            (feedback_id,)
        )
        
        if not feedback_exists:
            return {
                "code": 404,
                "message": "反馈不存在"
            }
        
        # 验证状态值是否有效
        valid_statuses = ['pending', 'processing', 'resolved', 'rejected']
        if request.status not in valid_statuses:
            return {
                "code": 400,
                "message": f"无效的状态值，必须是以下之一: {', '.join(valid_statuses)}"
            }
        
        # 更新反馈状态和回复
        if request.admin_reply:
            execute_update(
                """UPDATE user_feedback 
                   SET status = %s, admin_reply = %s, replied_at = NOW() 
                   WHERE feedback_id = %s""",
                (request.status, request.admin_reply, feedback_id)
            )
        else:
            execute_update(
                "UPDATE user_feedback SET status = %s WHERE feedback_id = %s",
                (request.status, feedback_id)
            )
        
        # 记录操作日志 (如果提供了管理员ID)
        if current_admin_id:
            try:
                execute_update(
                    """INSERT INTO operation_logs (admin_id, operation_type, operation_desc, created_at) 
                       VALUES (%s, %s, %s, NOW())""", 
                    (current_admin_id, "更新", f"管理员{current_admin_id}更新系统功能反馈{feedback_id}状态为{request.status}")
                )
            except Exception as log_error:
                # 仅记录日志错误，不影响主流程
                print(f"记录操作日志失败: {str(log_error)}")
        
        return {
            "code": 200,
            "message": "更新反馈状态成功"
        }
    except Exception as e:
        # 记录错误日志
        print(f"更新反馈状态失败: {str(e)}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"更新反馈状态失败: {str(e)}") 