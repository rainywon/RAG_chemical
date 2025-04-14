from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel, Field, validator, ValidationError
from typing import Optional
from database import execute_query, execute_update
from datetime import datetime
import logging
import sys
import traceback
import json

# 配置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 创建控制台处理器
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# 创建格式化器
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# 添加处理器到logger
logger.addHandler(console_handler)

# 初始化 APIRouter 实例，用于定义路由
router = APIRouter()

# 定义请求体的模型
class FeedbackRequest(BaseModel):
    rating: int = Field(..., description="评分，范围1-5")
    feedback: str = Field(..., description="反馈内容")
    feedback_option: str = Field(..., description="反馈选项：内容不准确, 回答不完整, 与问题不相关, 其他问题")
    message: str = Field(..., description="AI回答内容")
    question: str = Field(..., description="用户问题")

    @validator('rating')
    def validate_rating(cls, v):
        if v < 1 or v > 5:
            raise ValueError('评分必须在1-5之间')
        return v

    @validator('feedback_option')
    def validate_feedback_option(cls, v):
        valid_options = ['内容不准确', '回答不完整', '与问题不相关', '其他问题']
        if v not in valid_options:
            raise ValueError(f'反馈选项必须是以下之一: {", ".join(valid_options)}')
        return v

# 获取当前用户信息
async def get_current_user(request: Request):
    try:
        # 从Authorization头获取token
        auth_header = request.headers.get('Authorization')
        
        if not auth_header or not auth_header.startswith('Bearer '):
            raise HTTPException(status_code=401, detail="无效的认证信息")
        
        token = auth_header.split(' ')[1]
        
        # 查询admin_tokens表
        admin_result = execute_query(
            "SELECT admin_id FROM admin_tokens WHERE token = %s AND is_valid = 1 AND expire_at > NOW()",
            (token,)
        )
        
        if admin_result:
            return {
                "id": admin_result[0]['admin_id'],
                "role": "admin"
            }
        
        # 查询user_tokens表
        user_result = execute_query(
            "SELECT user_id FROM user_tokens WHERE token = %s AND is_valid = 1 AND expire_at > NOW()",
            (token,)
        )
        
        if user_result:
            return {
                "id": user_result[0]['user_id'],
                "role": "user"
            }
        
        raise HTTPException(status_code=401, detail="无效的token")
    except Exception as e:
        logger.error(f"Error in get_current_user: {str(e)}")
        logger.error(f"Error traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=401, detail=str(e))

# 记录操作日志
async def log_operation(user_info: dict, operation_type: str, operation_desc: str, request: Request = None):
    try:
        # 获取IP地址和用户代理
        ip_address = request.client.host if request else None
        user_agent = request.headers.get('user-agent') if request else None
        
        # 根据用户角色获取用户信息
        if user_info['role'] == 'admin':
            admin_result = execute_query(
                "SELECT full_name FROM admins WHERE admin_id = %s",
                (user_info['id'],)
            )
            if admin_result:
                user_name = admin_result[0]['full_name']
            else:
                return
        else:
            user_result = execute_query(
                "SELECT mobile FROM users WHERE user_id = %s",
                (user_info['id'],)
            )
            if user_result:
                user_name = f"用户{user_result[0]['mobile']}"
            else:
                return
        
        # 创建操作日志记录
        if user_info['role'] == 'admin':
            execute_update(
                """INSERT INTO operation_logs (admin_id, operation_type, operation_desc, ip_address, user_agent, created_at) 
                   VALUES (%s, %s, %s, %s, %s, NOW())""",
                (user_info['id'], operation_type, f"{user_name}{operation_desc}", ip_address, user_agent)
            )
        else:
            execute_update(
                """INSERT INTO operation_logs (user_id, operation_type, operation_desc, ip_address, user_agent, created_at) 
                   VALUES (%s, %s, %s, %s, %s, NOW())""",
                (user_info['id'], operation_type, f"{user_name}{operation_desc}", ip_address, user_agent)
            )
    except Exception as e:
        # 记录错误但不中断主要流程
        logger.error(f"记录操作日志失败: {str(e)}")

# 提交反馈的路由
@router.post("/submit-feedback/")
async def submit_feedback(
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    try:
        # 直接从请求中获取原始请求体
        body = await request.body()
        
        # 尝试解析JSON
        try:
            request_data = json.loads(body)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            raise HTTPException(status_code=400, detail="Invalid JSON in request body")
        
        # 手动验证请求数据
        try:
            feedback_request = FeedbackRequest(**request_data)
        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            logger.error(f"Validation error details: {e.errors()}")
            raise HTTPException(status_code=422, detail=str(e))

        # 记录操作日志
        await log_operation(
            current_user,
            "提交反馈",
            f"提交了评分{feedback_request.rating}的反馈，选项：{feedback_request.feedback_option}",
            request
        )

        # 插入反馈数据到数据库中
        query = """INSERT INTO content_feedbacks 
                  (user_id, rating, feedback, feedback_option, message, question, created_at, status) 
                  VALUES (%s, %s, %s, %s, %s, %s, NOW(), 'pending')"""
        params = (
            current_user['id'],
            feedback_request.rating,
            feedback_request.feedback,
            feedback_request.feedback_option,
            feedback_request.message,
            feedback_request.question
        )
        
        feedback_id = execute_update(query, params)
        logger.info(f"Feedback inserted successfully with ID: {feedback_id}")

        # 返回成功的响应
        return {"code": 200, "message": "反馈提交成功", "feedback_id": feedback_id}

    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        logger.error(f"Validation error details: {e.errors()}")
        logger.error(f"Validation error traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        logger.error(f"Value error: {str(e)}")
        logger.error(f"Value error traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(f"Unexpected error traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

# 获取用户反馈列表的路由（需要管理员权限）
@router.get("/feedbacks/")
async def get_feedbacks(
    request: Request,
    status: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    current_user: dict = Depends(get_current_user)
):
    try:
        # 检查用户权限
        if current_user['role'] != 'admin':
            raise HTTPException(status_code=403, detail="需要管理员权限")
        
        # 构建查询条件
        query_conditions = []
        params = []
        
        if status:
            valid_statuses = ["pending", "processing", "resolved", "rejected"]
            if status not in valid_statuses:
                return {"code": 400, "message": f"无效的状态，支持的状态有: {', '.join(valid_statuses)}"}
            
            query_conditions.append("status = %s")
            params.append(status)
        
        # 构建完整查询
        query = """SELECT f.*, u.nickname, u.mobile 
                   FROM content_feedbacks f 
                   LEFT JOIN users u ON f.user_id = u.user_id"""
        
        if query_conditions:
            query += " WHERE " + " AND ".join(query_conditions)
        
        query += " ORDER BY f.created_at DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])
        
        # 执行查询
        feedbacks = execute_query(query, params)
        
        # 记录操作日志
        await log_operation(
            current_user,
            "查询反馈",
            f"查询了反馈列表，状态：{status if status else '全部'}",
            request
        )
        
        # 构建响应
        return {
            "code": 200,
            "message": "获取反馈列表成功",
            "data": {
                "total": len(feedbacks),
                "feedbacks": feedbacks
            }
        }
    
    except Exception as e:
        logger.error(f"获取反馈列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
