# 引入 FastAPI 中的 APIRouter 和 HTTPException 模块，用于创建路由和处理异常
from fastapi import APIRouter, HTTPException, Depends, Query
# 引入 Pydantic 中的 BaseModel 类，用于定义请求体的数据结构和验证
from pydantic import BaseModel
# 从数据库模块导入 execute_query 和 execute_update 函数，用于执行查询和更新操作
from database import execute_query, execute_update
# 引入时间模块处理日期范围
from datetime import datetime
# 引入 typing 模块中的 Optional 类型
from typing import Optional
# 引入管理员认证依赖函数
from routers.login import get_current_admin
# 引入密码加密模块
import hashlib

# 初始化 APIRouter 实例，用于定义路由
router = APIRouter()

# 定义管理员密码修改请求模型
class PasswordChangeRequest(BaseModel):
    admin_id: int  # 管理员ID
    old_password: str  # 当前密码
    new_password: str  # 新密码

# 定义管理员信息更新请求模型
class AdminProfileUpdateRequest(BaseModel):
    admin_id: int  # 管理员ID
    full_name: str  # 姓名
    email: Optional[str] = None  # 邮箱

# 修改管理员密码接口
@router.post("/admin/admins/change-password", tags=["管理员管理"])
async def change_password(request: PasswordChangeRequest):
    """
    修改管理员密码
    """
    try:
        # 验证管理员是否存在
        admin = execute_query(
            """SELECT * FROM admins WHERE admin_id = %s""", 
            (request.admin_id,)
        )
        
        if not admin:
            return {
                "code": 404,
                "message": "管理员不存在"
            }
        
        # 验证旧密码
        old_password_hash = hashlib.md5(request.old_password.encode()).hexdigest()
        
        if admin[0]['password'] != old_password_hash:
            return {
                "code": 400,
                "message": "当前密码错误"
            }
        
        # 加密新密码
        new_password_hash = hashlib.md5(request.new_password.encode()).hexdigest()
        
        # 更新密码
        execute_update(
            """UPDATE admins SET password = %s, updated_at = NOW() WHERE admin_id = %s""", 
            (new_password_hash, request.admin_id)
        )
        
        # 记录操作日志
        execute_update(
            """INSERT INTO operation_logs (admin_id, operation_type, operation_desc, created_at) 
               VALUES (%s, %s, %s, NOW())""", 
            (request.admin_id, "update", f"管理员{request.admin_id}修改了密码")
        )
        
        return {
            "code": 200,
            "message": "密码修改成功"
        }
    except Exception as e:
        # 记录错误日志
        print(f"修改密码失败: {str(e)}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"修改密码失败: {str(e)}")

# 更新管理员个人信息接口
@router.put("/admin/admins/profile", tags=["管理员管理"])
async def update_admin_profile(request: AdminProfileUpdateRequest):
    """
    更新管理员个人信息
    """
    try:
        # 验证管理员是否存在
        admin = execute_query(
            """SELECT * FROM admins WHERE admin_id = %s""", 
            (request.admin_id,)
        )
        
        if not admin:
            return {
                "code": 404,
                "message": "管理员不存在"
            }
        
        # 更新管理员信息
        execute_update(
            """UPDATE admins SET full_name = %s, email = %s, updated_at = NOW() WHERE admin_id = %s""", 
            (request.full_name, request.email, request.admin_id)
        )
        
        # 记录操作日志
        execute_update(
            """INSERT INTO operation_logs (admin_id, operation_type, operation_desc, created_at) 
               VALUES (%s, %s, %s, NOW())""", 
            (request.admin_id, "update", f"管理员{request.admin_id}更新了个人信息")
        )
        
        # 获取更新后的管理员信息
        updated_admin = execute_query(
            """SELECT admin_id, phone_number, full_name, role, email, status, 
                   last_login_time, created_at, updated_at 
               FROM admins 
               WHERE admin_id = %s""", 
            (request.admin_id,)
        )
        
        # 处理日期时间格式
        admin_data = updated_admin[0]
        admin_data['last_login_time'] = admin_data['last_login_time'].strftime("%Y-%m-%d %H:%M:%S") if admin_data['last_login_time'] else None
        admin_data['created_at'] = admin_data['created_at'].strftime("%Y-%m-%d %H:%M:%S") if admin_data['created_at'] else None
        admin_data['updated_at'] = admin_data['updated_at'].strftime("%Y-%m-%d %H:%M:%S") if admin_data['updated_at'] else None
        
        return {
            "code": 200,
            "message": "个人信息更新成功",
            "data": admin_data
        }
    except Exception as e:
        # 记录错误日志
        print(f"更新个人信息失败: {str(e)}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"更新个人信息失败: {str(e)}")
