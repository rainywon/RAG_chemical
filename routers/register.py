# 引入 FastAPI 中的 APIRouter 和 HTTPException 模块，用于创建路由和处理异常
from fastapi import APIRouter, HTTPException
# 引入 Pydantic 中的 BaseModel 类，用于定义请求体的数据结构和验证
from pydantic import BaseModel
# 从数据库模块导入 execute_query 和 execute_update 函数，用于执行查询和更新操作
from database import execute_query, execute_update
# 导入密码哈希处理模块
import hashlib

# 初始化 APIRouter 实例，用于定义路由
router = APIRouter()


# 定义请求体的模型，使用 Pydantic 的 BaseModel 来验证请求的数据
class RegisterRequest(BaseModel):
    # 定义注册所需的字段
    mobile: str
    code: str  # 验证码
    password: str  # 密码
    confirm_password: str  # 确认密码


# 创建一个 POST 请求的路由，路径为 "/register/"
@router.post("/register/")
# 异步处理函数，接收 RegisterRequest 类型的请求体
async def register(request: RegisterRequest):
    try:
        # 验证请求参数
        if not request.mobile or len(request.mobile) != 11:
            return {"code": 400, "message": "请提供有效的11位手机号码"}
        
        if request.password != request.confirm_password:
            return {"code": 400, "message": "两次输入的密码不一致"}
        
        if len(request.password) < 6:
            return {"code": 400, "message": "密码长度不能少于6位"}

        if len(request.code) != 6:
            return {"code": 400, "message": "验证码必须是6位数字"}
        
        # 检查手机号是否已被注册
        user_exists = execute_query("""SELECT * FROM users WHERE mobile = %s LIMIT 1""", (request.mobile,))
        if user_exists:
            return {"code": 400, "message": "该手机号已注册，请直接登录"}
        
        # 验证验证码是否正确且未被使用，且是否在有效期内（15分钟内）
        result = execute_query(
            """SELECT * FROM sms_codes WHERE mobile = %s AND code = %s AND type = 'register' AND is_used = FALSE AND created_at > NOW() - INTERVAL 15 MINUTE ORDER BY created_at DESC LIMIT 1""",
            (request.mobile, request.code))

        # 如果没有找到符合条件的验证码，返回错误信息
        if not result:
            return {"code": 400, "message": "验证码错误或已过期"}

        # 获取验证码记录
        code_record = result[0]
        # 标记该验证码为已使用
        execute_update("""UPDATE sms_codes SET is_used = TRUE WHERE id = %s""", (code_record['id'],))

        # 对密码进行哈希处理
        hashed_password = hashlib.md5(request.password.encode()).hexdigest()
        
        # 插入新用户记录
        execute_update("""INSERT INTO users (mobile, password) VALUES (%s, %s)""", 
                      (request.mobile, hashed_password))
        
        # 获取新插入的用户 ID
        user_id = execute_query("SELECT LAST_INSERT_ID()")[0]['LAST_INSERT_ID()']
        
        # 生成token（简单示例，实际应使用JWT等更安全的方式）
        token = f"{user_id}_{hashlib.sha256(f'{user_id}_{request.mobile}'.encode()).hexdigest()}"
        
        # 返回成功注册的响应，并返回用户ID和token
        return {
            "code": 200, 
            "message": "注册成功", 
            "user_id": user_id,
            "token": token
        }

    # 捕获异常并返回 HTTP 500 错误，附带错误信息
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 