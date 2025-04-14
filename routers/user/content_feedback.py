from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from database import execute_query, execute_update
 
 # 初始化 APIRouter 实例，用于定义路由
router = APIRouter()
 
 # 定义请求体的模型，使用 Pydantic 的 BaseModel 来验证请求的数据
class FeedbackRequest(BaseModel):
     user_id: int  # 用户ID
     feedback_type: str  # 反馈内容
     feedback_content: str  # 反馈选项
 
 # 创建一个 POST 请求的路由，路径为 "/submit-content_feedback/"
@router.post("/submit-content_feedback/")
 # 异步处理函数，接收 FeedbackRequest 类型的请求体
async def submit_feedback(request: FeedbackRequest):
     try:
         # 插入反馈数据到数据库中
         query = """INSERT INTO user_feedback (user_id, feedback_type, feedback_content) 
                    VALUES (%s, %s, %s)"""
         params = (request.user_id, request.feedback_type, request.feedback_content)
         execute_update(query, params)
 
         # 返回成功的响应
         return {"code": 200, "message": "反馈提交成功"}
 
     except Exception as e:
         # 捕获异常并返回 HTTP 500 错误，附带错误信息
         raise HTTPException(status_code=500, detail=str(e))