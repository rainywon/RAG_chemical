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
from datetime import datetime, timedelta
from fastapi import Request
import jwt
from config import Config

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

# 定义管理员登录请求的模型
class AdminLoginRequest(BaseModel):
    mobile: str  # 管理员用户名
    password: str  # 管理员密码

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
        
        # 查询用户是否存在及状态
        user_status_check = execute_query("""SELECT user_id, status FROM users WHERE mobile = %s LIMIT 1""", (request.mobile,))
        
        # 检查用户状态，如果已被禁用则拒绝登录
        if user_status_check and user_status_check[0]['status'] == 0:
            return {"code": 403, "message": "账号已被禁用，请联系管理员"}
        
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
                    """INSERT INTO users (mobile, theme_preference, register_time, status) VALUES (%s, 'light', NOW(), 1)""", 
                    (request.mobile,))
            else:
                # 密码登录模式下，用户不存在则返回错误
                return {"code": 400, "message": "用户不存在，请先注册"}
        else:
            # 如果用户已存在，获取用户 ID
            user_id = user_result[0]['user_id']
            
            # 再次检查用户状态，确保用户没有被禁用
            if user_result[0]['status'] == 0:
                return {"code": 403, "message": "账号已被禁用，请联系管理员"}
                
            # 更新最后登录时间和登录次数
            execute_update(
                """UPDATE users SET last_login_time = NOW() WHERE user_id = %s""", 
                (user_id,))

        # 生成token并存入数据库
        token = str(uuid.uuid4())
        # 设置token过期时间（7天后）
        expire_at = datetime.datetime.now() + datetime.timedelta(days=7)
        
        # 先检查该用户是否已有token记录
        existing_token = execute_query(
            """SELECT id FROM user_tokens WHERE user_id = %s AND is_valid = 1 LIMIT 1""",
            (user_id,)
        )
        
        if existing_token:
            # 如果已有token，则更新token信息
            execute_update(
                """UPDATE user_tokens SET token = %s, created_at = NOW(), expire_at = %s 
                   WHERE user_id = %s AND is_valid = 1""",
                (token, expire_at, user_id)
            )
        else:
            # 如果没有token，则创建新的token记录
            execute_update(
                """INSERT INTO user_tokens (user_id, token, created_at, expire_at, is_valid) 
                   VALUES (%s, %s, NOW(), %s, 1)""",
                (user_id, token, expire_at)
            )
        
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

# 创建管理员登录路由
@router.post("/admin_login/")
async def admin_login(admin_login_data: AdminLoginRequest):
    try:
        # 验证管理员账号和密码
        hashed_password = hashlib.md5(admin_login_data.password.encode()).hexdigest()
        
        # 查询管理员表中是否存在该管理员并验证密码
        # 使用phone_number字段匹配前端传来的mobile
        admin = execute_query(
            """SELECT * FROM admins WHERE phone_number = %s AND password = %s AND status = 1 LIMIT 1""", 
            (admin_login_data.mobile, hashed_password)
        )
        
        if not admin:
            return {"code": 401, "message": "管理员不存在或密码错误或已被禁用"}
        
        admin = admin[0]
        
        # 更新管理员最后登录时间
        execute_update(
            """UPDATE admins SET last_login_time = NOW() WHERE admin_id = %s""", 
            (admin['admin_id'],)
        )
        
        # 生成token
        expires = datetime.now() + timedelta(days=1)
        token = str(uuid.uuid4())
        
        # 查询是否已有token
        token_result = execute_query(
            """SELECT id FROM admin_tokens WHERE admin_id = %s AND is_valid = 1 LIMIT 1""", 
            (admin['admin_id'],)
        )
        
        # 更新或新增token
        if token_result:
            execute_update(
                """UPDATE admin_tokens SET token = %s, created_at = NOW(), expire_at = %s 
                   WHERE admin_id = %s AND is_valid = 1""", 
                (token, expires, admin['admin_id'])
            )
        else:
            execute_update(
                """INSERT INTO admin_tokens (admin_id, token, created_at, expire_at, is_valid) 
                   VALUES (%s, %s, NOW(), %s, 1)""", 
                (admin['admin_id'], token, expires)
            )
        
        # 记录管理员登录操作
        try:
            admin_name = admin['full_name'] if 'full_name' in admin else f"管理员{admin['admin_id']}"
            
            execute_update(
                """INSERT INTO operation_logs (admin_id, operation_type, operation_desc, created_at) 
                   VALUES (%s, %s, %s, NOW())""", 
                (admin['admin_id'], "登录", f"管理员[{admin_name}]登录系统")
            )
        except Exception as log_error:
            # 仅记录日志错误，不影响主流程
            print(f"记录操作日志失败: {str(log_error)}")
        
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
    except Exception as e:
        return {"code": 500, "message": f"登录失败: {str(e)}"}

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
            print(f"记录管理员查看信息操作日志失败: {str(e)}")
        
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
            print(f"记录操作日志失败: {str(log_error)}")
        
        return {"code": 200, "message": "退出登录成功"}
    except Exception as e:
        return {"code": 500, "message": f"退出登录失败: {str(e)}"}
