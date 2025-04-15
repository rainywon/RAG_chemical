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
# 引入可选类型
from typing import Optional
# 统一使用datetime导入方式，避免命名冲突
from datetime import datetime, timedelta
from fastapi import Request
import jwt
from config import Config
import logging

# 初始化 APIRouter 实例，用于定义路由
router = APIRouter()

# 创建 HTTPBearer 实例用于处理 Bearer token
security = HTTPBearer()

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 定义请求体的模型，使用 Pydantic 的 BaseModel 来验证请求的数据
class LoginRequest(BaseModel):
    # 定义登录请求所需的字段
    mobile: str
    mode: str  # 登录模式: 'code'为验证码登录, 'password'为密码登录
    code: Optional[str] = None  # 验证码，验证码登录时需要
    password: Optional[str] = None  # 密码，密码登录时需要

# 定义管理员登录请求的模型
class AdminLoginRequest(BaseModel):
    username: str  # 管理员用户名
    password: str  # 管理员密码

# 更新chat_sessions表中的token
async def update_chat_sessions_token(old_token: str, new_token: str):
    try:
        # 更新chat_sessions表中的token
        affected_rows = execute_update(
            """UPDATE chat_sessions SET token = %s WHERE token = %s""",
            (new_token, old_token)
        )
        logger.debug(f"已更新 {affected_rows} 条聊天会话记录的token")
        return affected_rows
    except Exception as e:
        logger.error(f"更新聊天会话token失败: {str(e)}")
        # 不抛出异常，因为这不应该中断登录流程
        return 0

# 创建一个 POST 请求的路由，路径为 "/login/"
@router.post("/login/")
# 异步处理函数，接收 LoginRequest 类型的请求体
async def login(request: LoginRequest):
    """
    用户登录接口，支持验证码和密码两种登录方式
    处理用户验证、token生成和更新、会话关联等逻辑
    """
    try:
        logger.debug(f"开始登录处理，登录模式: {request.mode}, 手机号: {request.mobile}")
        
        # 第一步: 验证请求参数
        await validate_login_request(request)
        
        # 第二步: 验证用户凭据(验证码或密码)
        user_id = await validate_user_credentials(request)
        
        # 第三步: 处理用户token(检查、更新或创建)
        token, is_new_token = await manage_user_token(user_id)
        
        # 返回登录成功的响应
        logger.debug("登录过程完成，准备返回成功响应")
        return {
            "code": 200, 
            "message": "登录成功", 
            "data": {
                "user_id": user_id,
                "token": token
            }
        }
    except HTTPException:
        # 直接向上抛出HTTP异常，保持原状态码
        raise
    except Exception as e:
        # 捕获并记录其他所有异常
        logger.error(f"登录过程发生异常: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"登录过程发生错误: {str(e)}")

async def validate_login_request(request: LoginRequest):
    """验证登录请求的基本参数"""
    # 验证登录模式
    if request.mode not in ['code', 'password']:
        logger.warning(f"登录模式错误: {request.mode}")
        raise HTTPException(status_code=400, detail="登录模式不正确，只支持code或password")
    
    # 根据登录模式验证必要参数
    if request.mode == 'code' and not request.code:
        logger.warning("验证码模式下，验证码为空")
        raise HTTPException(status_code=400, detail="验证码登录模式下，验证码不能为空")
    
    if request.mode == 'password' and not request.password:
        logger.warning("密码模式下，密码为空")
        raise HTTPException(status_code=400, detail="密码登录模式下，密码不能为空")
    
    # 检查用户状态(是否被禁用)
    user_status_check = execute_query(
        """SELECT user_id, status FROM users WHERE mobile = %s LIMIT 1""", 
        (request.mobile,)
    )
    logger.debug(f"用户状态检查结果: {user_status_check}")
    
    if user_status_check and user_status_check[0]['status'] == 0:
        logger.warning("用户已被禁用")
        raise HTTPException(status_code=403, detail="账号已被禁用，请联系管理员")

async def validate_user_credentials(request: LoginRequest):
    """验证用户登录凭据，返回用户ID"""
    if request.mode == 'code':
        # 验证码登录模式
        return await validate_code_login(request)
    else:
        # 密码登录模式
        return await validate_password_login(request)

async def validate_code_login(request: LoginRequest):
    """处理验证码登录逻辑"""
    logger.debug("验证码登录模式")
    # 验证验证码是否正确且未被使用，且是否在有效期内
    result = execute_query(
        """SELECT * FROM verification_codes 
           WHERE mobile = %s AND code = %s AND is_used = 0 
           AND purpose = 'login' AND expire_at > NOW() 
           ORDER BY created_at DESC LIMIT 1""",
        (request.mobile, request.code)
    )
    
    logger.debug(f"验证码验证结果: {result}")
    if not result:
        raise HTTPException(status_code=400, detail="验证码错误或已过期")
    
    # 标记该验证码为已使用
    code_record = result[0]
    execute_update(
        """UPDATE verification_codes SET is_used = 1 WHERE id = %s""", 
        (code_record['id'],)
    )
    
    # 查询用户是否已注册
    user_result = execute_query(
        """SELECT * FROM users WHERE mobile = %s LIMIT 1""", 
        (request.mobile,)
    )
    
    if not user_result:
        # 用户不存在，自动注册
        logger.info("用户不存在，进行自动注册")
        user_id = execute_update(
            """INSERT INTO users (mobile, theme_preference, register_time, status) 
               VALUES (%s, 'light', NOW(), 1)""", 
            (request.mobile,)
        )
        logger.info(f"用户自动注册成功，ID: {user_id}")
        return user_id
    else:
        # 用户已存在，更新登录时间
        user_id = user_result[0]['user_id']
        execute_update(
            """UPDATE users SET last_login_time = NOW() WHERE user_id = %s""", 
            (user_id,)
        )
        logger.debug(f"用户存在，ID: {user_id}，已更新登录时间")
        return user_id

async def validate_password_login(request: LoginRequest):
    """处理密码登录逻辑"""
    logger.debug("密码登录模式")
    # 对密码进行哈希处理
    hashed_password = hashlib.md5(request.password.encode()).hexdigest()
    
    # 查询用户是否存在并验证密码
    user_result = execute_query(
        """SELECT * FROM users WHERE mobile = %s AND password = %s AND status = 1 LIMIT 1""",
        (request.mobile, hashed_password)
    )
    logger.debug(f"密码验证结果: {bool(user_result)}")
    
    if not user_result:
        raise HTTPException(status_code=400, detail="手机号或密码错误，或账号已被禁用")
    
    # 更新用户最后登录时间
    user_id = user_result[0]['user_id']
    execute_update(
        """UPDATE users SET last_login_time = NOW() WHERE user_id = %s""", 
        (user_id,)
    )
    logger.debug(f"密码验证成功，用户ID: {user_id}，已更新登录时间")
    return user_id

async def manage_user_token(user_id: int):
    """
    管理用户token: 检查现有token，必要时创建新token
    返回: (token, is_new_token)
    """
    # 检查是否有现有token
    existing_token_record = execute_query(
        """SELECT token, is_valid, expire_at 
           FROM user_tokens 
           WHERE user_id = %s 
           ORDER BY expire_at DESC LIMIT 1""",
        (user_id,)
    )
    logger.debug(f"现有token记录检查结果: {existing_token_record}")
    
    # 设置token过期时间（7天后）
    expire_at = datetime.now() + timedelta(days=7)
    
    # 如果有现有token且未过期，直接使用
    if existing_token_record and existing_token_record[0]['expire_at'] > datetime.now():
        token = existing_token_record[0]['token']
        
        # 检查is_valid状态，如果非1则更新为1
        if existing_token_record[0]['is_valid'] != 1:
            logger.debug(f"token未过期但is_valid为{existing_token_record[0]['is_valid']}，更新为有效状态")
            execute_update(
                """UPDATE user_tokens SET is_valid = 1 
                   WHERE user_id = %s AND token = %s""",
                (user_id, token)
            )
        
        logger.debug(f"现有token未过期，继续使用: {token}")
        return token, False
    
    # 生成新token
    new_token = str(uuid.uuid4())
    logger.debug(f"生成新token: {new_token}, 过期时间: {expire_at}")
    
    # 如果有旧token，更新它并关联的会话
    if existing_token_record:
        old_token = existing_token_record[0]['token']
        logger.debug(f"更新现有token: {old_token} -> {new_token}")
        
        # 更新token记录
        execute_update(
            """UPDATE user_tokens 
               SET token = %s, created_at = NOW(), expire_at = %s, is_valid = 1 
               WHERE user_id = %s AND token = %s""",
            (new_token, expire_at, user_id, old_token)
        )
        
        # 更新关联的聊天会话
        await update_chat_sessions_token(old_token, new_token)
    else:
        # 创建新token记录
        logger.debug(f"创建新token记录: {new_token}")
        execute_update(
            """INSERT INTO user_tokens (user_id, token, created_at, expire_at, is_valid) 
               VALUES (%s, %s, NOW(), %s, 1)""",
            (user_id, new_token, expire_at)
        )
    
    return new_token, True

# 创建管理员登录路由
@router.post("/admin_login/")
async def admin_login(admin_login_data: AdminLoginRequest):
    """
    管理员登录接口，验证管理员账号和密码，并处理token的生成和管理
    """
    try:
        # 第一步：验证管理员凭据并获取管理员信息
        admin = await validate_admin_credentials(admin_login_data)
        
        # 第二步：更新管理员最后登录时间
        execute_update(
            """UPDATE admins SET last_login_time = NOW() WHERE admin_id = %s""", 
            (admin['admin_id'],)
        )
        
        # 第三步：处理管理员token
        token, is_new_token = await manage_admin_token(admin['admin_id'])
        
        # 第四步：记录管理员登录操作
        await log_admin_login(admin)
        
        # 返回登录成功响应
        return {
            "code": 200,
            "data": {
                "token": token,
                "adminId": admin['admin_id'],
                "username": admin['phone_number'],
                "name": admin.get('full_name', admin['phone_number']),
                "avatar": admin.get('avatar_url', ''),
                "role": admin['role']
            },
            "message": "登录成功"
        }
    except HTTPException:
        # 直接向上抛出HTTP异常，保持原状态码
        raise
    except Exception as e:
        # 捕获并记录其他所有异常
        logger.error(f"管理员登录失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"登录失败: {str(e)}")

async def validate_admin_credentials(admin_login_data: AdminLoginRequest):
    """验证管理员登录凭据，返回管理员信息"""
    # 对密码进行哈希处理
    hashed_password = hashlib.md5(admin_login_data.password.encode()).hexdigest()
    logger.debug(f"验证管理员凭据: {admin_login_data.username}")
    
    # 查询管理员表中是否存在该管理员并验证密码
    admin = execute_query(
        """SELECT * FROM admins 
           WHERE phone_number = %s AND password = %s AND status = 1 
           LIMIT 1""", 
        (admin_login_data.username, hashed_password)
    )
    
    if not admin:
        logger.warning(f"管理员验证失败: {admin_login_data.username}")
        raise HTTPException(
            status_code=401, 
            detail="管理员不存在或密码错误或已被禁用"
        )
    
    logger.debug(f"管理员验证成功: ID={admin[0]['admin_id']}")
    return admin[0]

async def manage_admin_token(admin_id: int):
    """
    管理管理员token: 检查现有token，必要时创建新token
    返回: (token, is_new_token)
    """
    # 检查是否有现有token
    existing_token_record = execute_query(
        """SELECT token, is_valid, expire_at 
           FROM admin_tokens 
           WHERE admin_id = %s 
           ORDER BY expire_at DESC LIMIT 1""",
        (admin_id,)
    )
    logger.debug(f"管理员现有token记录检查结果: {existing_token_record}")
    
    # 设置token过期时间（1天后）
    expire_at = datetime.now() + timedelta(days=1)
    
    # 如果有现有token且未过期，直接使用
    if existing_token_record and existing_token_record[0]['expire_at'] > datetime.now():
        token = existing_token_record[0]['token']
        
        # 检查is_valid状态，如果非1则更新为1
        if existing_token_record[0]['is_valid'] != 1:
            logger.debug(f"管理员token未过期但is_valid为{existing_token_record[0]['is_valid']}，更新为有效状态")
            execute_update(
                """UPDATE admin_tokens SET is_valid = 1 
                   WHERE admin_id = %s AND token = %s""",
                (admin_id, token)
            )
        
        logger.debug(f"管理员现有token未过期，继续使用: {token}")
        return token, False
    
    # 生成新token
    new_token = str(uuid.uuid4())
    logger.debug(f"生成新管理员token: {new_token}, 过期时间: {expire_at}")
    
    # 如果有旧token，更新它并更新关联的会话
    if existing_token_record:
        old_token = existing_token_record[0]['token']
        logger.debug(f"更新现有管理员token: {old_token} -> {new_token}")
        
        # 更新token记录
        execute_update(
            """UPDATE admin_tokens 
               SET token = %s, created_at = NOW(), expire_at = %s, is_valid = 1 
               WHERE admin_id = %s AND token = %s""",
            (new_token, expire_at, admin_id, old_token)
        )
        
        # 更新关联的聊天会话
        await update_chat_sessions_token(old_token, new_token)
    else:
        # 创建新token记录
        logger.debug(f"创建新管理员token记录: {new_token}")
        execute_update(
            """INSERT INTO admin_tokens 
               (admin_id, token, created_at, expire_at, is_valid) 
               VALUES (%s, %s, NOW(), %s, 1)""",
            (admin_id, new_token, expire_at)
        )
    
    return new_token, True

async def log_admin_login(admin):
    """记录管理员登录操作日志"""
    try:
        admin_name = admin.get('full_name', f"管理员{admin['admin_id']}")
        
        execute_update(
            """INSERT INTO operation_logs 
               (admin_id, operation_type, operation_desc, created_at) 
               VALUES (%s, %s, %s, NOW())""", 
            (admin['admin_id'], "登录", f"管理员[{admin_name}]登录系统")
        )
        logger.debug(f"记录管理员登录操作成功: {admin_name}")
    except Exception as e:
        # 仅记录日志错误，不影响主流程
        logger.error(f"记录管理员登录操作日志失败: {str(e)}")

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

# 获取当前管理员的依赖函数
async def get_current_admin(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        # 从 Authorization header 中获取 token
        token = credentials.credentials
        
        # 验证token是否有效
        token_result = execute_query(
            """SELECT admin_id FROM admin_tokens 
               WHERE token = %s AND is_valid = 1 AND expire_at > NOW() LIMIT 1""", 
            (token,)
        )
        
        if not token_result:
            raise HTTPException(status_code=401, detail="无效的管理员令牌或令牌已过期")
        
        # 获取管理员ID
        admin_id = token_result[0]['admin_id']
        
        # 验证管理员是否存在
        admin = execute_query("SELECT * FROM admins WHERE admin_id = %s AND status = 1", (admin_id,))
        if not admin:
            raise HTTPException(status_code=401, detail="管理员不存在或已被禁用")
            
        return admin_id
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"管理员认证失败: {str(e)}")

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

# 获取管理员信息的路由
@router.get("/admin_info/")
async def admin_info(request: Request, admin_id: int = Depends(get_current_admin)):
    try:
        # 查询管理员信息
        admin_data = execute_query(
            """SELECT admin_id, phone_number, full_name, role, email, status,
               last_login_time, created_at, updated_at
               FROM admins
               WHERE admin_id = %s
               LIMIT 1""", 
            (admin_id,)
        )
        
        if not admin_data:
            raise HTTPException(status_code=404, detail="管理员不存在")
        
        admin = admin_data[0]
        
        # 记录管理员查看信息的操作
        try:
            execute_update(
                """INSERT INTO operation_logs 
                   (admin_id, operation_type, operation_desc, created_at) 
                   VALUES (%s, %s, %s, NOW())""",
                (
                    admin_id,
                    "查看管理员信息",
                    f"管理员 {admin['full_name']} 查看了自己的信息"
                )
            )
        except Exception as e:
            logger.error(f"记录管理员查看信息操作日志失败: {str(e)}")
        
        # 返回管理员信息
        return {
            "success": True,
            "message": "获取管理员信息成功",
            "data": {
                "admin_id": admin["admin_id"],
                "phone_number": admin["phone_number"],
                "full_name": admin["full_name"],
                "role": admin["role"],
                "status": admin["status"],
                "last_login_time": admin["last_login_time"],
                "email": admin["email"]
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取管理员信息失败: {str(e)}")

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

# 管理员退出登录
@router.post("/admin_logout/")
async def admin_logout(admin_id: int = Depends(get_current_admin), credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        # 获取当前token
        token = credentials.credentials
        
        # 使token失效
        execute_update(
            """UPDATE admin_tokens SET is_valid = 0 WHERE token = %s""", 
            (token,)
        )
        
        # 记录管理员退出登录操作
        try:
            admin_info = execute_query(
                """SELECT full_name FROM admins WHERE admin_id = %s""", 
                (admin_id,)
            )
            
            admin_name = admin_info[0]['full_name'] if admin_info else f"管理员{admin_id}"
            
            execute_update(
                """INSERT INTO operation_logs (admin_id, operation_type, operation_desc, created_at) 
                   VALUES (%s, %s, %s, NOW())""", 
                (admin_id, "退出", f"管理员[{admin_name}]退出登录")
            )
        except Exception as log_error:
            # 仅记录日志错误，不影响主流程
            logger.error(f"记录操作日志失败: {str(log_error)}")
        
        return {"code": 200, "message": "退出登录成功"}
    except Exception as e:
        return {"code": 500, "message": f"退出登录失败: {str(e)}"}
