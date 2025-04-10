# 引入 FastAPI 中的 APIRouter 和 HTTPException 模块，用于创建路由和处理异常
from fastapi import APIRouter, HTTPException, Query
# 引入 Pydantic 中的 BaseModel 类，用于定义请求体的数据结构和验证
from pydantic import BaseModel
# 引入 requests 模块，用于发送 HTTP 请求
import requests
# 从配置文件中导入 APPKEY 和 APPSECRET
from config import APPKEY, APPSECRET

# 初始化 APIRouter 实例，用于定义路由
router = APIRouter()

# 定义一个短信报告请求模型
class SmsReportRequest(BaseModel):
    smsid: str

# 创建一个 GET 请求的路由，路径为 "/sms_report/"
@router.get("/sms_report/")
async def get_sms_report(smsid: str = Query(..., description="短信ID")):
    """
    获取短信报告请求
    """
    try:
        # API请求URL
        url = "http://api.guoyangyun.com/api/sms/smsReport.htm"
        
        # 请求参数
        params = {
            "appkey": APPKEY,
            "appsecret": APPSECRET,
            "smsid": smsid
        }
        
        # 发送GET请求
        response = requests.get(url, params=params)
        
        # 检查请求是否成功
        response.raise_for_status()
        
        # 尝试解析JSON响应
        try:
            return response.json()
        except:
            # 如果不是JSON，返回文本响应
            return {"status": "success", "data": response.text}
        
    except requests.RequestException as err:
        raise HTTPException(status_code=500, detail=f"短信报告请求失败: {err}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 