# 引入 uvicorn，用于启动 FastAPI 应用
import uvicorn
# 引入 FastAPI 类，用于创建 FastAPI 应用实例
from fastapi import FastAPI
# 引入 CORSMiddleware，用于处理跨源资源共享（CORS）问题
from fastapi.middleware.cors import CORSMiddleware
# 从 routers 目录中导入不同模块的路由（query、sms、login）
from routers.query import router as query_router
from routers.sms import router as sms_router
from routers.login import router as login_router
from routers.submit_feedback import router as submit_feedback_router
from routers.sms_report import router as sms_report_router
from routers.register import router as register_router
from routers.chat_history import router as chat_history_router
# 创建 FastAPI 应用实例
app = FastAPI()

# 配置 CORS 中间件，允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源的请求
    allow_credentials=True,  # 允许携带凭证（如 Cookies）
    allow_methods=["*"],  # 允许所有 HTTP 方法（如 GET、POST）
    allow_headers=["*"],  # 允许所有请求头
)

# 将各个模块的路由包含到主应用中
app.include_router(query_router)  # 包含 ai生成内容 路由
app.include_router(sms_router)  # 包含 验证码发送 路由
app.include_router(login_router)  # 包含 登录请求 路由
app.include_router(submit_feedback_router)  # 包含 提交反馈 路由
app.include_router(sms_report_router)  # 包含 短信报告 路由
app.include_router(register_router)  # 包含 注册请求 路由
app.include_router(chat_history_router)  # 包含 聊天历史 路由
if __name__ == '__main__':
    # 启动应用并监听 127.0.0.1:8000 端口，启用自动重载功能
    uvicorn.run("main:app",
                host="127.0.0.1",
                port=8000,
                reload=False,  # 关闭自动重载
                )
