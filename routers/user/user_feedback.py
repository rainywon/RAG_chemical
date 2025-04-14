from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional
from database import execute_query, execute_update
from routers.user.login import get_current_user

# 初始化 APIRouter 实例，用于定义路由
router = APIRouter()

# 定义请求体的模型，使用 Pydantic 的 BaseModel 来验证请求的数据
class FeedbackRequest(BaseModel):
    feedback_type: str = Field(..., description="反馈类型：suggestion, bug, content, other")
    feedback_content: str = Field(..., description="反馈内容")

# 创建一个 POST 请求的路由，路径为 "/submit-feedback/"
@router.post("/submit-user-feedback/")
# 异步处理函数，接收 FeedbackRequest 类型的请求体，以及用户身份验证
async def submit_feedback(request: FeedbackRequest, user_id: Optional[int] = None):
    try:
        # 验证反馈类型
        valid_types = ["suggestion", "bug", "content", "other"]
        if request.feedback_type not in valid_types:
            return {"code": 400, "message": f"无效的反馈类型，支持的类型有: {', '.join(valid_types)}"}
        
        # 验证反馈内容
        if not request.feedback_content or len(request.feedback_content.strip()) < 5:
            return {"code": 400, "message": "反馈内容不能少于5个字符"}
        
        # 验证评分（如果提供）
        if request.rating is not None and (request.rating < 1 or request.rating > 5):
            return {"code": 400, "message": "评分范围必须在1-5之间"}

        # 插入反馈数据到数据库中
        query = """INSERT INTO user_feedback (user_id, feedback_type, feedback_content, created_at, status) 
                   VALUES (%s, %s, %s, NOW(), 'pending')"""
        params = (user_id, request.feedback_type, request.feedback_content)
        
        feedback_id = execute_update(query, params)

        # 返回成功的响应
        return {"code": 200, "message": "反馈提交成功", "feedback_id": feedback_id}

    except Exception as e:
        # 捕获异常并返回 HTTP 500 错误，附带错误信息
        raise HTTPException(status_code=500, detail=str(e))

# 获取用户反馈列表的路由（需要管理员权限）
@router.get("/feedbacks/")
async def get_feedbacks(status: Optional[str] = None, limit: int = 20, offset: int = 0):
    try:
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
                   FROM user_feedback f 
                   LEFT JOIN users u ON f.user_id = u.user_id"""
        
        if query_conditions:
            query += " WHERE " + " AND ".join(query_conditions)
        
        query += " ORDER BY f.created_at DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])
        
        # 执行查询
        feedbacks = execute_query(query, params)
        
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
        raise HTTPException(status_code=500, detail=str(e))
