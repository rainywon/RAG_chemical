# 引入 uvicorn，用于启动 FastAPI 应用
import uvicorn
# 引入 FastAPI 类，用于创建 FastAPI 应用实例
from fastapi import FastAPI
# 引入 CORSMiddleware，用于处理跨源资源共享（CORS）问题
from fastapi.middleware.cors import CORSMiddleware
# 从 routers 目录中导入不同模块的路由
from routers.user.query import router as query_router
from routers.user.sms import router as sms_router
from routers.user.login import router as login_router
from routers.user.submit_feedback import router as submit_feedback_router
from routers.user.sms_report import router as sms_report_router
from routers.user.register import router as register_router
from routers.user.user_feedback import router as user_feedback_router
from routers.user.content_feedback import router as content_feedback_router
from routers.user.system import router as system_router
from routers.user.chat_history import router as chat_history_router
from routers.user.SaftyFiles import router as safety_files_router
from routers.user.EmergencyFiles import router as emergency_files_router
from routers.admin.monitor.ConversationStat import router as conversation_stat_router
from routers.admin.monitor.UserActivity import router as user_activity_router
from routers.admin.users.UserManagement import router as user_management_router
from routers.admin.users.LoginHistory import router as login_history_router
from routers.admin.admins.AdminManagement import router as admin_management_router
from routers.admin.admins.OperationLogs import router as operation_logs_router
from routers.admin.feedback.SystemFunctionFeedback import router as system_function_feedback_router
from routers.admin.feedback.AIContentFeedback import router as ai_content_feedback_router
from routers.admin.settings.SystemParams import router as system_params_router
from routers.admin.content.CategoryManager import router as category_manager_router
from routers.admin.content.DocumentManager import router as document_manager_router
from routers.admin.content.EmergencyPlanManager import router as emergency_plan_manager_router
from routers.admin.AdminDashboard import router as admin_dashboard_router
from routers.admin.AdminLayout import router as admin_layout_router
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
app.include_router(user_feedback_router)  # 包含 用户反馈 路由
app.include_router(content_feedback_router)  # 包含 内容反馈 路由
app.include_router(system_router)  # 包含 系统 路由
app.include_router(chat_history_router)  # 包含 聊天历史 路由
app.include_router(safety_files_router)  # 包含 安全文件 路由
app.include_router(conversation_stat_router)  # 包含 对话统计 路由
app.include_router(user_activity_router)  # 包含 用户活跃度 路由
app.include_router(user_management_router)  # 包含 用户管理 路由
app.include_router(login_history_router)  # 包含 登录历史 路由
app.include_router(admin_management_router)  # 包含 管理员管理 路由
app.include_router(operation_logs_router)  # 包含 操作日志 路由
app.include_router(system_function_feedback_router)  # 包含 系统功能反馈 路由
app.include_router(ai_content_feedback_router)  # 包含 AI内容反馈 路由
app.include_router(system_params_router)  # 包含 系统参数设置 路由
app.include_router(category_manager_router)  # 包含 知识库文件管理 路由
app.include_router(document_manager_router)  # 包含 安全资料库管理 路由
app.include_router(emergency_plan_manager_router)  # 包含 应急预案管理 路由
app.include_router(admin_dashboard_router)  # 包含 管理员仪表盘 路由
app.include_router(admin_layout_router)  # 包含 管理员布局 路由
app.include_router(emergency_files_router)  # 包含 应急文件 路由
if __name__ == '__main__':
    # 启动应用并监听 127.0.0.1:8000 端口，启用自动重载功能
    uvicorn.run("main:app",
                host="127.0.0.1",
                port=8000,
                reload=False,  # 关闭自动重载
                )
