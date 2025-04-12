# 引入 FastAPI 中的 APIRouter 和 HTTPException 模块，用于创建路由和处理异常
from fastapi import APIRouter, HTTPException, Depends, Query
# 引入 Pydantic 中的 BaseModel 类，用于定义请求体的数据结构和验证
from pydantic import BaseModel
# 从数据库模块导入 execute_query 和 execute_update 函数，用于执行查询和更新操作
from database import execute_query, execute_update
# 引入时间模块处理日期范围
from datetime import datetime, timedelta
# 引入 typing 模块中的 Optional 和 List 类型
from typing import Optional, List
# 引入管理员认证依赖函数
from routers.login import get_current_admin
# 引入密码加密模块
import hashlib

# 初始化 APIRouter 实例，用于定义路由
router = APIRouter()

# 定义管理员请求模型
class AdminCreateRequest(BaseModel):
    phone_number: str  # 手机号
    full_name: str  # 姓名
    email: Optional[str] = None  # 邮箱
    role: str  # 角色 admin/operator
    password: str  # 密码
    current_admin_id: Optional[int] = None  # 当前操作的管理员ID

# 定义管理员更新请求模型
class AdminUpdateRequest(BaseModel):
    admin_id: int  # 管理员ID
    full_name: str  # 姓名
    email: Optional[str] = None  # 邮箱
    role: str  # 角色 admin/operator
    current_admin_id: Optional[int] = None  # 当前操作的管理员ID

# 定义管理员状态更新请求模型
class AdminStatusRequest(BaseModel):
    admin_id: int  # 管理员ID
    status: int  # 状态: 0-禁用, 1-正常
    current_admin_id: Optional[int] = None  # 当前操作的管理员ID

# 定义管理员密码更新请求模型
class AdminPasswordRequest(BaseModel):
    admin_id: int  # 管理员ID
    old_password: str  # 旧密码
    new_password: str  # 新密码
    current_admin_id: Optional[int] = None  # 当前操作的管理员ID

# 获取管理员列表接口
@router.get("/admin/admins", tags=["管理员管理"])
async def get_admin_list(
    page: int = Query(1, description="页码"),
    page_size: int = Query(10, description="每页数量"),
    phone_number: Optional[str] = Query(None, description="手机号筛选"),
    full_name: Optional[str] = Query(None, description="姓名筛选"),
    role: Optional[str] = Query(None, description="角色筛选"),
    admin_id: Optional[int] = Query(None, description="当前管理员ID")
):
    """
    获取管理员列表，支持分页和筛选
    """
    try:
        # 计算偏移量
        offset = (page - 1) * page_size
        
        # 构建基础查询SQL
        query = """
            SELECT 
                admin_id, phone_number, full_name, role, email, status, 
                last_login_time, created_at, updated_at
            FROM 
                admins
            WHERE 1=1
        """
        params = []
        
        # 添加筛选条件
        if phone_number:
            query += " AND phone_number LIKE %s"
            params.append(f"%{phone_number}%")
        
        if full_name:
            query += " AND full_name LIKE %s"
            params.append(f"%{full_name}%")
        
        if role:
            query += " AND role = %s"
            params.append(role)
        
        # 查询符合条件的总记录数
        count_query = f"SELECT COUNT(*) as count FROM ({query}) as filtered_admins"
        count_result = execute_query(count_query, tuple(params))
        total_count = count_result[0]['count'] if count_result else 0
        
        # 添加排序和分页
        query += " ORDER BY admin_id ASC LIMIT %s OFFSET %s"
        params.append(page_size)
        params.append(offset)
        
        # 查询管理员列表
        admin_list = execute_query(query, tuple(params))
        
        # 处理日期时间格式
        for admin in admin_list:
            admin['last_login_time'] = admin['last_login_time'].strftime("%Y-%m-%d %H:%M:%S") if admin['last_login_time'] else None
            admin['created_at'] = admin['created_at'].strftime("%Y-%m-%d %H:%M:%S") if admin['created_at'] else None
            admin['updated_at'] = admin['updated_at'].strftime("%Y-%m-%d %H:%M:%S") if admin['updated_at'] else None
        
        # 记录操作日志 (如果提供了管理员ID)
        if admin_id:
            try:
                execute_update(
                    """INSERT INTO operation_logs (admin_id, operation_type, operation_desc, created_at) 
                       VALUES (%s, %s, %s, NOW())""", 
                    (admin_id, "query", f"管理员{admin_id}查询管理员列表")
                )
            except Exception as log_error:
                # 仅记录日志错误，不影响主流程
                print(f"记录操作日志失败: {str(log_error)}")
        
        return {
            "code": 200,
            "message": "获取管理员列表成功",
            "data": {
                "admins": admin_list,
                "total": total_count
            }
        }
    except Exception as e:
        # 记录错误日志
        print(f"获取管理员列表失败: {str(e)}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"获取管理员列表失败: {str(e)}")

# 创建管理员接口
@router.post("/admin/admins", tags=["管理员管理"])
async def create_admin(request: AdminCreateRequest):
    """
    创建新管理员
    """
    try:
        # 验证手机号是否已存在
        admin_check = execute_query(
            """SELECT * FROM admins WHERE phone_number = %s""", 
            (request.phone_number,)
        )
        
        if admin_check:
            return {
                "code": 400,
                "message": "该手机号已被使用"
            }
        
        # 密码加密
        hashed_password = hashlib.md5(request.password.encode()).hexdigest()
        
        # 插入新管理员
        execute_update(
            """INSERT INTO admins (phone_number, password, full_name, role, email, status, created_at, updated_at) 
               VALUES (%s, %s, %s, %s, %s, 1, NOW(), NOW())""", 
            (request.phone_number, hashed_password, request.full_name, request.role, request.email)
        )
        
        # 记录操作日志
        if request.current_admin_id:
            execute_update(
                """INSERT INTO operation_logs (admin_id, operation_type, operation_desc, created_at) 
                   VALUES (%s, %s, %s, NOW())""", 
                (request.current_admin_id, "create", f"管理员{request.current_admin_id}创建了新管理员{request.phone_number}")
            )
        
        return {
            "code": 200,
            "message": "创建管理员成功"
        }
    except Exception as e:
        # 记录错误日志
        print(f"创建管理员失败: {str(e)}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"创建管理员失败: {str(e)}")

# 更新管理员信息接口
@router.put("/admin/admins/{admin_id}", tags=["管理员管理"])
async def update_admin(admin_id: int, request: AdminUpdateRequest):
    """
    更新管理员信息
    """
    try:
        # 验证管理员是否存在
        admin_check = execute_query(
            """SELECT * FROM admins WHERE admin_id = %s""", 
            (admin_id,)
        )
        
        if not admin_check:
            return {
                "code": 404,
                "message": "管理员不存在"
            }
        
        # 更新管理员信息
        execute_update(
            """UPDATE admins SET full_name = %s, email = %s, role = %s, updated_at = NOW() 
               WHERE admin_id = %s""", 
            (request.full_name, request.email, request.role, admin_id)
        )
        
        # 记录操作日志
        if request.current_admin_id:
            execute_update(
                """INSERT INTO operation_logs (admin_id, operation_type, operation_desc, created_at) 
                   VALUES (%s, %s, %s, NOW())""", 
                (request.current_admin_id, "update", f"管理员{request.current_admin_id}更新了管理员{admin_id}的信息")
            )
        
        return {
            "code": 200,
            "message": "更新管理员信息成功"
        }
    except Exception as e:
        # 记录错误日志
        print(f"更新管理员信息失败: {str(e)}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"更新管理员信息失败: {str(e)}")

# 更改管理员状态接口
@router.post("/admin/admins/status", tags=["管理员管理"])
async def update_admin_status(request: AdminStatusRequest):
    """
    更新管理员账户状态
    """
    try:
        # 验证管理员是否存在
        admin_check = execute_query(
            """SELECT * FROM admins WHERE admin_id = %s""", 
            (request.admin_id,)
        )
        
        if not admin_check:
            return {
                "code": 404,
                "message": "管理员不存在"
            }
        
        # 更新管理员状态
        execute_update(
            """UPDATE admins SET status = %s, updated_at = NOW() WHERE admin_id = %s""", 
            (request.status, request.admin_id)
        )
        
        # 记录操作日志
        operation_type = "启用管理员" if request.status == 1 else "禁用管理员"
        
        if request.current_admin_id:
            execute_update(
                """INSERT INTO operation_logs (admin_id, operation_type, operation_desc, created_at) 
                   VALUES (%s, %s, %s, NOW())""", 
                (request.current_admin_id, "update", f"管理员{request.current_admin_id}{operation_type}{request.admin_id}")
            )
        
        return {
            "code": 200,
            "message": "更新管理员状态成功"
        }
    except Exception as e:
        # 记录错误日志
        print(f"更新管理员状态失败: {str(e)}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"更新管理员状态失败: {str(e)}")

# 修改管理员密码接口
@router.post("/admin/admins/change-password", tags=["管理员管理"])
async def change_admin_password(request: AdminPasswordRequest):
    """
    修改管理员密码
    """
    try:
        # 验证管理员是否存在
        admin_check = execute_query(
            """SELECT * FROM admins WHERE admin_id = %s""", 
            (request.admin_id,)
        )
        
        if not admin_check:
            return {
                "code": 404,
                "message": "管理员不存在"
            }
        
        # 验证旧密码
        hashed_old_password = hashlib.md5(request.old_password.encode()).hexdigest()
        
        password_check = execute_query(
            """SELECT * FROM admins WHERE admin_id = %s AND password = %s""", 
            (request.admin_id, hashed_old_password)
        )
        
        if not password_check:
            return {
                "code": 400,
                "message": "旧密码错误"
            }
        
        # 更新密码
        hashed_new_password = hashlib.md5(request.new_password.encode()).hexdigest()
        
        execute_update(
            """UPDATE admins SET password = %s, updated_at = NOW() WHERE admin_id = %s""", 
            (hashed_new_password, request.admin_id)
        )
        
        # 记录操作日志
        if request.current_admin_id:
            execute_update(
                """INSERT INTO operation_logs (admin_id, operation_type, operation_desc, created_at) 
                   VALUES (%s, %s, %s, NOW())""", 
                (request.current_admin_id, "update", f"管理员{request.current_admin_id}修改了密码")
            )
        
        return {
            "code": 200,
            "message": "修改密码成功"
        }
    except Exception as e:
        # 记录错误日志
        print(f"修改密码失败: {str(e)}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"修改密码失败: {str(e)}")
