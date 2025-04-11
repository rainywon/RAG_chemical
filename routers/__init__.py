# 导入各个路由模块
from .login import router as login_router
from .sms import router as sms_router
from .query import router as query_router
from .user_feedback import router as feedback_router
from .register import router as register_router

# 导出所有路由
routers = [
    login_router,
    sms_router,
    query_router,
    feedback_router,
    register_router,
] 