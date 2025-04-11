# 引入 FastAPI 中的 APIRouter 和 HTTPException 模块，用于创建路由和处理异常
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
# 引入 Pydantic 中的 BaseModel 类，用于定义请求体的数据结构和验证
from pydantic import BaseModel
# 从数据库模块导入 execute_query 和 execute_update 函数，用于执行查询和更新操作
from database import execute_query, execute_update
# 导入密码哈希处理模块
import hashlib
# 导入生成令牌所需的模块
import uuid
import datetime
# 引入可选类型
from typing import Optional

# 初始化 APIRouter 实例，用于定义路由
router = APIRouter()

# 创建 HTTPBearer 实例用于处理 Bearer token
security = HTTPBearer()

# 定义请求体的模型，使用 Pydantic 的 BaseModel 来验证请求的数据
class LoginRequest(BaseModel):
    # 定义登录请求所需的字段
    mobile: str
    mode: str  # 登录模式: 'code'为验证码登录, 'password'为密码登录
    code: Optional[str] = None  # 验证码，验证码登录时需要
    password: Optional[str] = None  # 密码，密码登录时需要


# 创建一个 POST 请求的路由，路径为 "/login/"
@router.post("/login/")
# 异步处理函数，接收 LoginRequest 类型的请求体
async def login(request: LoginRequest):
    try:
        # 验证请求参数
        if request.mode not in ['code', 'password']:
            return {"code": 400, "message": "登录模式不正确，只支持code或password"}
        
        if request.mode == 'code' and not request.code:
            return {"code": 400, "message": "验证码登录模式下，验证码不能为空"}
        
        if request.mode == 'password' and not request.password:
            return {"code": 400, "message": "密码登录模式下，密码不能为空"}
        
        # 根据登录模式选择不同的验证方式
        if request.mode == 'code':
            # 验证码登录
            # 执行数据库查询，验证验证码是否正确且未被使用，且是否在有效期内
            result = execute_query(
                """SELECT * FROM verification_codes WHERE mobile = %s AND code = %s AND is_used = 0 
                   AND purpose = 'login' AND expire_at > NOW() ORDER BY created_at DESC LIMIT 1""",
                (request.mobile, request.code))

            # 如果没有找到符合条件的验证码，返回错误信息
            if not result:
                return {"code": 400, "message": "验证码错误或已过期"}

            # 获取验证码记录
            code_record = result[0]
            # 标记该验证码为已使用
            execute_update("""UPDATE verification_codes SET is_used = 1 WHERE id = %s""", (code_record['id'],))
        else:
            # 密码登录
            # 查询用户是否存在并验证密码
            # 对密码进行哈希处理再比较
            hashed_password = hashlib.md5(request.password.encode()).hexdigest()
            user_result = execute_query(
                """SELECT * FROM users WHERE mobile = %s AND password = %s LIMIT 1""", 
                (request.mobile, hashed_password))
            
            if not user_result:
                return {"code": 400, "message": "手机号或密码错误"}

        # 查询用户是否已经注册
        user_result = execute_query("""SELECT * FROM users WHERE mobile = %s LIMIT 1""", (request.mobile,))

        if not user_result:
            # 如果用户不存在，则进行自动注册（仅在验证码登录模式下）
            if request.mode == 'code':
                # 使用默认主题偏好设置创建用户
                user_id = execute_update(
                    """INSERT INTO users (mobile, theme_preference, register_time) VALUES (%s, 'light', NOW())""", 
                    (request.mobile,))
            else:
                # 密码登录模式下，用户不存在则返回错误
                return {"code": 400, "message": "用户不存在，请先注册"}
        else:
            # 如果用户已存在，获取用户 ID
            user_id = user_result[0]['user_id']
            # 更新最后登录时间和登录次数
            execute_update(
                """UPDATE users SET last_login_time = NOW(), login_count = login_count + 1 WHERE user_id = %s""", 
                (user_id,))

        # 生成token并存入数据库
        token = str(uuid.uuid4())
        # 设置token过期时间（7天后）
        expire_at = datetime.datetime.now() + datetime.timedelta(days=7)
        
        # 记录用户登录令牌
        execute_update(
            """INSERT INTO user_tokens (user_id, token, created_at, expire_at, is_valid) 
               VALUES (%s, %s, NOW(), %s, 1)""",
            (user_id, token, expire_at))
        
        # 返回成功登录的响应，并返回用户ID和token
        return {
            "code": 200, 
            "message": "登录成功", 
            "user_id": user_id,
            "token": token
        }

    # 捕获异常并返回 HTTP 500 错误，附带错误信息
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 获取当前用户的依赖函数
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        # 从 Authorization header 中获取 token
        token = credentials.credentials
        
        # 验证token是否有效
        token_result = execute_query(
            """SELECT * FROM user_tokens 
               WHERE token = %s AND is_valid = 1 AND expire_at > NOW() LIMIT 1""", 
            (token,)
        )
        
        if not token_result:
            raise HTTPException(status_code=401, detail="无效的令牌或令牌已过期")
        
        # 获取用户ID
        user_id = token_result[0]['user_id']
        
        # 验证用户是否存在
        user = execute_query("SELECT * FROM users WHERE user_id = %s AND status = 1", (user_id,))
        if not user:
            raise HTTPException(status_code=401, detail="用户不存在或已被禁用")
            
        return user_id
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"认证失败: {str(e)}")

# 获取用户信息的路由
@router.get("/user/info/")
async def get_user_info(user_id: int = Depends(get_current_user)):
    try:
        # 查询用户信息
        user_info = execute_query("SELECT user_id, mobile, theme_preference FROM users WHERE user_id = %s", (user_id,))
        
        if not user_info:
            raise HTTPException(status_code=404, detail="未找到用户信息")
        
        return {
            "code": 200,
            "message": "获取用户信息成功",
            "data": user_info[0]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 用户登出的路由
@router.post("/logout/")
async def logout(user_id: int = Depends(get_current_user), credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        # 获取当前token
        token = credentials.credentials
        
        # 将token设为无效
        execute_update(
            """UPDATE user_tokens SET is_valid = 0 WHERE token = %s""", 
            (token,)
        )
        
        return {
            "code": 200,
            "message": "登出成功"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
