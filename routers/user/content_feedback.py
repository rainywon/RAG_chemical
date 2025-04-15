from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel
from database import execute_query, execute_update
from typing import Optional
from datetime import datetime
import logging

# 初始化 APIRouter 实例，用于定义路由
router = APIRouter()

# 初始化日志记录器
logger = logging.getLogger(__name__)

# 定义请求体的模型
class FeedbackRequest(BaseModel):
    feedback_type: str  # 反馈类型
    feedback_content: str  # 反馈内容

# 获取当前用户信息
async def get_current_user(request: Request):
    try:
        # 从Authorization头获取token
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="无效的认证信息")
        
        token = auth_header.split(" ")[1]
        
        # 查询admin_tokens表
        admin_result = execute_query(
            "SELECT admin_id FROM admin_tokens WHERE token = %s AND is_valid = 1 AND expire_at > NOW()",
            (token,)
        )
        
        if admin_result:
            return {
                "id": admin_result[0]["admin_id"],
                "role": "admin"
            }
        
        # 查询user_tokens表
        user_result = execute_query(
            "SELECT user_id FROM user_tokens WHERE token = %s AND is_valid = 1 AND expire_at > NOW()",
            (token,)
        )
        
        if user_result:
            return {
                "id": user_result[0]["user_id"],
                "role": "user"
            }
        
        raise HTTPException(status_code=401, detail="无效的token")
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))

# 记录操作日志
async def log_operation(user_info: dict, operation_type: str, operation_desc: str, request: Request = None):
    try:
        # 获取IP地址和用户代理
        ip_address = request.client.host if request else None
        user_agent = request.headers.get("user-agent") if request else None
        
        # 根据用户角色获取用户信息
        if user_info["role"] == "admin":
            admin_result = execute_query(
                "SELECT full_name FROM admins WHERE admin_id = %s",
                (user_info["id"],)
            )
            if admin_result:
                user_name = admin_result[0]["full_name"]
            else:
                return
        else:
            user_result = execute_query(
                "SELECT mobile FROM users WHERE user_id = %s",
                (user_info["id"],)
            )
            if user_result:
                user_name = f"用户{user_result[0]['mobile']}"
            else:
                return
        
        # 创建操作日志记录
        if user_info["role"] == "admin":
            execute_update(
                """INSERT INTO operation_logs (admin_id, operation_type, operation_desc, ip_address, user_agent, created_at) 
                   VALUES (%s, %s, %s, %s, %s, NOW())""",
                (user_info["id"], operation_type, f"{user_name}{operation_desc}", ip_address, user_agent)
            )
        else:
            execute_update(
                """INSERT INTO operation_logs (user_id, operation_type, operation_desc, ip_address, user_agent, created_at) 
                   VALUES (%s, %s, %s, %s, %s, NOW())""",
                (user_info["id"], operation_type, f"{user_name}{operation_desc}", ip_address, user_agent)
            )
    except Exception as e:
        # 记录日志但不影响主流程
        logger.error(f"记录操作日志失败: {str(e)}")

# 提交反馈的路由
@router.post("/submit-content-feedback/")
async def submit_feedback(
    request: Request,
    feedback: FeedbackRequest,
    current_user: dict = Depends(get_current_user)
):
    try:
        # 记录操作日志
        await log_operation(
            current_user,
            "提交反馈",
            f"提交了{feedback.feedback_type}类型的反馈",
            request
        )

        # 插入反馈数据到数据库中
        query = """INSERT INTO user_feedback (user_id, feedback_type, feedback_content, created_at) 
                   VALUES (%s, %s, %s, NOW())"""
        params = (current_user["id"], feedback.feedback_type, feedback.feedback_content)
        execute_update(query, params)

        # 返回成功的响应
        return {"code": 200, "message": "反馈提交成功"}

    except Exception as e:
        # 捕获异常并返回 HTTP 500 错误，附带错误信息
        raise HTTPException(status_code=500, detail=str(e))
