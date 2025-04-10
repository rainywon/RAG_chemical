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

# 初始化一个 APIRouter 实例，用于定义路由
router = APIRouter()


# 定义一个 POST 请求的路由，路径为 "/send_sms/"
@router.post("/send_sms/")
# 异步处理函数，接收 mobile（手机号码）和 type（验证码类型）作为请求体参数
async def send_sms(mobile: str = Body(..., embed=True), type: Optional[str] = Body("login", embed=True)):
    try:
        # 验证手机号格式
        if not mobile or len(mobile) != 11:
            return {"code": 400, "message": "请提供有效的11位手机号码"}
        
        # 对于注册类型，检查手机号是否已注册
        if type == "register":
            user_exists = execute_query("""SELECT id FROM users WHERE mobile = %s LIMIT 1""", (mobile,))
            if user_exists:
                return {"code": 400, "message": "该手机号已注册，请直接登录"}
        
        # 生成一个 6 位的随机验证码
        code = ''.join([str(random.randint(0, 9)) for _ in range(6)])
        # 设置短信模板中的参数，**code** 表示验证码，**minute** 表示有效时间（15分钟）
        param = f"code:{code}"  # 使用动态生成的验证码
        # 构造发送短信的请求数据
        data = {
            "phone_number": mobile,  # 目标手机号码
            "content": param,  # 短信模板的参数
            "template_id":"CST_ptdie100"
        }
        # 设置请求头，包含 API 密钥等信息
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": "APPCODE " + APPCODE  # 使用配置的 APPCODE 进行身份验证
        }
        
        # 清理过期的验证码数据：删除超过 15 分钟的验证码记录
        execute_update("""DELETE FROM sms_codes WHERE mobile = %s AND created_at < NOW() - INTERVAL 15 MINUTE""",
                       (mobile,))
                       
        # 限制验证码发送频率，一分钟内只能发送一次
        recent_code = execute_query(
            """SELECT * FROM sms_codes WHERE mobile = %s AND created_at > NOW() - INTERVAL 1 MINUTE ORDER BY created_at DESC LIMIT 1""",
            (mobile,))
        if recent_code:
            return {"code": 429, "message": "发送过于频繁，请稍后再试"}
                   
        # 插入新生成的验证码到数据库，带上类型
        execute_update("""INSERT INTO sms_codes (mobile, code, type) VALUES (%s, %s, %s)""", (mobile, code, type))

        # 发送短信请求
        response = requests.post(URL, headers=headers, data=data)
        # 检查请求是否成功，若失败则抛出异常
        response.raise_for_status()

        # 返回成功响应
        return {"code": 200, "message": "验证码发送成功"}

    # 捕获请求异常，返回 500 错误并附带错误信息
    except requests.RequestException as err:
        raise HTTPException(status_code=500, detail=f"短信发送请求失败: {err}")
    # 捕获其他类型的异常，返回 500 错误并附带错误信息
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
