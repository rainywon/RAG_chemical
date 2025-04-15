from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from database import execute_query, execute_update
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 初始化 APIRouter 实例，用于定义路由
router = APIRouter()

# 创建 HTTPBearer 实例用于处理 Bearer token
security = HTTPBearer()

# 定义请求体的模型，使用 Pydantic 的 BaseModel 来验证请求的数据
class FeedbackRequest(BaseModel):
    rating: int  # 星级评分
    feedback: str  # 反馈内容
    feedback_option: str  # 反馈选项
    message: str  # AI 回答内容
    question: str

# 获取当前用户的依赖函数
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        # 从 Authorization header 中获取 token
        token = credentials.credentials
        logger.debug(f"接收到的token: {token}")
        
        # 验证token是否有效
        token_result = execute_query(
            """SELECT * FROM user_tokens 
               WHERE token = %s AND is_valid = 1 AND expire_at > NOW() LIMIT 1""", 
            (token,)
        )
        
        if not token_result:
            logger.error("无效的令牌或令牌已过期")
            raise HTTPException(status_code=401, detail="无效的令牌或令牌已过期")
        
        # 获取用户ID
        user_id = token_result[0]['user_id']
        logger.debug(f"已验证用户ID: {user_id}")
        
        # 验证用户是否存在
        user = execute_query("SELECT * FROM users WHERE user_id = %s AND status = 1", (user_id,))
        if not user:
            logger.error("用户不存在或已被禁用")
            raise HTTPException(status_code=401, detail="用户不存在或已被禁用")
            
        return user_id
    except Exception as e:
        logger.error(f"认证失败: {str(e)}")
        raise HTTPException(status_code=401, detail=f"认证失败: {str(e)}")

# 创建一个 POST 请求的路由，路径为 "/submit-feedback/"
@router.post("/submit-feedback/")
# 异步处理函数，接收 FeedbackRequest 类型的请求体
async def submit_feedback(request: FeedbackRequest, user_id: int = Depends(get_current_user)):
    try:
        logger.debug(f"收到反馈提交请求，用户ID: {user_id}, 评分: {request.rating}")
        
        # 验证请求参数
        if request.rating < 1 or request.rating > 5:
            logger.warning(f"无效的评分: {request.rating}")
            raise HTTPException(status_code=400, detail="评分必须在1到5之间")
            
        if len(request.feedback) > 2000:
            logger.warning("反馈内容过长")
            raise HTTPException(status_code=400, detail="反馈内容过长")
        
        # 插入反馈数据到数据库中
        query = """INSERT INTO content_feedbacks 
                   (user_id, rating, feedback, feedback_option, message, question, created_at, status) 
                   VALUES (%s, %s, %s, %s, %s, %s, NOW(), 'pending')"""
        params = (user_id, request.rating, request.feedback, request.feedback_option, request.message, request.question)
        
        logger.debug(f"执行SQL: {query}, 参数: {user_id}, {request.rating}, {request.feedback_option}")
        execute_update(query, params)
        
        # 记录操作日志
        try:
            log_query = """INSERT INTO operation_logs 
                          (user_id, operation_type, operation_desc, created_at) 
                          VALUES (%s, %s, %s, NOW())"""
            log_params = (user_id, "提交反馈", f"用户{user_id}提交了{request.feedback_option}类型的反馈")
            execute_update(log_query, log_params)
        except Exception as log_error:
            # 仅记录日志错误，不影响主流程
            logger.error(f"记录操作日志失败: {str(log_error)}")

        # 返回成功的响应
        logger.debug("反馈提交成功")
        return {"code": 200, "message": "反馈提交成功"}

    except HTTPException:
        # 直接向上抛出HTTP异常，保持原状态码
        raise
    except Exception as e:
        # 捕获异常并记录详细日志
        logger.error(f"反馈提交失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"反馈提交失败: {str(e)}")
