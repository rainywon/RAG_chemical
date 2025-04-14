# 引入 FastAPI 中的 APIRouter、HTTPException 和 Body，用于创建路由、处理异常和获取请求体
from fastapi import APIRouter, HTTPException, Body
# 引入 random 模块，用于生成随机验证码
import random
# 引入 requests 模块，用于发送 HTTP 请求
import requests
# 从配置文件中导入 URL、APPCODE、SMS_SIGN_ID 和 TEMPLATE_ID，用于短信服务的配置
from config import URL, APPCODE, SMS_SIGN_ID, TEMPLATE_ID,APPKEY,APPSECRET
# 从数据库模块导入 execute_update 和 execute_query 函数，用于执行数据库更新和查询操作
from database import execute_update, execute_query
# 导入可选类型
from typing import Optional
# 引入Pydantic中的BaseModel类，用于定义请求体的数据结构和验证
from pydantic import BaseModel
# 引入时间模块，用于处理时间
import datetime

# 初始化一个 APIRouter 实例，用于定义路由
router = APIRouter()

# 定义请求体的模型，使用Pydantic的BaseModel来验证请求的数据
class SmsRequest(BaseModel):
    # 手机号码字段，必须提供
    mobile: str
    # 验证码用途，可选字段，默认为登录
    purpose: str = 'login'

# 定义一个 POST 请求的路由，路径为 "/send_sms/"
@router.post("/send_sms/")
# 异步处理函数，接收 SmsRequest 类型的请求体
async def send_sms(request: SmsRequest):
    try:
        # 验证手机号格式
        if not is_valid_mobile(request.mobile):
            return {"code": 400, "message": "无效的手机号码格式"}

        # 验证用途是否有效
        valid_purposes = ['login', 'register', 'reset_password']
        if request.purpose not in valid_purposes:
            return {"code": 400, "message": f"无效的验证码用途，有效用途为: {', '.join(valid_purposes)}"}

        # 检查是否存在1分钟内发送过的验证码
        recent_code = execute_query(
            """SELECT * FROM verification_codes 
               WHERE mobile = %s AND purpose = %s AND created_at > DATE_SUB(NOW(), INTERVAL 1 MINUTE) 
               LIMIT 1""",
            (request.mobile, request.purpose)
        )
        
        if recent_code:
            return {"code": 429, "message": "发送过于频繁，请稍后再试"}

        # 生成6位随机数字验证码
        verification_code = ''.join([str(random.randint(0, 9)) for _ in range(6)])
        
        # 设置验证码过期时间（15分钟后）
        expire_at = datetime.datetime.now() + datetime.timedelta(minutes=15)
        
        # 将验证码插入数据库
        execute_update(
            """INSERT INTO verification_codes (mobile, code, purpose, created_at, expire_at, is_used) 
               VALUES (%s, %s, %s, NOW(), %s, 0)""",
            (request.mobile, verification_code, request.purpose, expire_at)
        )

        # 构造发送短信的请求数据
        data = {
            "phone_number": request.mobile,
            "content": f"code:{verification_code}",
            "template_id": TEMPLATE_ID
        }
        
        # 设置请求头
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": "APPCODE " + APPCODE
        }
        
        # 发送短信请求
        response = requests.post(URL, headers=headers, data=data)
        response.raise_for_status()
        
        # 返回成功响应
        return {
            "code": 200, 
            "message": "验证码已发送"
        }

    except requests.RequestException as err:
        raise HTTPException(status_code=500, detail=f"短信发送请求失败: {err}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 验证手机号格式是否正确的函数
def is_valid_mobile(mobile):
    """
    验证手机号格式是否正确
    """
    # 简单验证，实际应根据需求调整
    return mobile.isdigit() and len(mobile) == 11 and mobile.startswith('1')

# 检查验证码路由，用于验证用户输入的验证码是否正确
@router.post("/verify_code/")
async def verify_code(mobile: str, code: str, purpose: str = 'login'):
    try:
        # 查询是否存在有效的验证码
        result = execute_query(
            """SELECT * FROM verification_codes 
               WHERE mobile = %s AND code = %s AND purpose = %s AND is_used = 0 
               AND expire_at > NOW() ORDER BY created_at DESC LIMIT 1""",
            (mobile, code, purpose)
        )
        
        if not result:
            return {"code": 400, "message": "验证码错误或已过期"}
        
        # 验证码正确，但不标记为已使用
        # 在实际登录或注册过程中再标记为已使用
        
        return {"code": 200, "message": "验证码正确"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
