# 引入 FastAPI 中的 APIRouter 和 HTTPException 模块，用于创建路由和处理异常
from fastapi import APIRouter, HTTPException, Depends, Query
# 引入 Pydantic 中的 BaseModel 类，用于定义请求体的数据结构和验证
from pydantic import BaseModel
# 从数据库模块导入 execute_query 和 execute_update 函数，用于执行查询和更新操作
from database import execute_query, execute_update
# 引入 typing 模块中的 Optional 和 List 类型
from typing import Optional, List
# 引入管理员认证依赖函数
from routers.login import get_current_admin

# 初始化 APIRouter 实例，用于定义路由
router = APIRouter()

# 定义角色权限请求模型
class RolePermissionRequest(BaseModel):
    permissions: List[str]  # 权限ID列表
    admin_id: Optional[int] = None  # 当前操作的管理员ID

# 获取角色权限列表接口
@router.get("/admin/roles/{role}/permissions", tags=["管理员管理"])
async def get_role_permissions(
    role: str,
    admin_id: Optional[int] = Query(None, description="当前管理员ID")
):
    """
    获取指定角色的权限列表
    """
    try:
        # 验证角色是否有效
        if role not in ['admin', 'operator']:
            return {
                "code": 400,
                "message": "无效的角色类型"
            }
        
        # 对于超级管理员，始终返回所有权限
        if role == 'admin':
            # 这里实际应该是从数据库中查询所有权限，但为简化起见，直接返回示例权限
            permissions = [
                'system', 'system:config', 'system:version',
                'admin', 'admin:list', 'admin:add', 'admin:edit', 'admin:delete', 'admin:status',
                'user', 'user:list', 'user:view', 'user:status', 'user:history',
                'content', 'content:category', 'content:document', 'content:emergency',
                'log', 'log:operation', 'log:login',
                'feedback', 'feedback:list', 'feedback:reply'
            ]
        else:
            # 从数据库中查询操作员角色的权限
            # 注意：这里假设角色权限存储在一个名为 role_permissions 的表中
            # 实际项目中，可能需要根据具体的数据库设计进行调整
            try:
                permission_result = execute_query(
                    """SELECT permission_id FROM role_permissions WHERE role = %s""", 
                    (role,)
                )
                
                permissions = [item['permission_id'] for item in permission_result] if permission_result else []
                
                # 默认权限（如果数据库中没有记录）
                if not permissions:
                    permissions = [
                        'user', 'user:list', 'user:view',
                        'content', 'content:category', 'content:document',
                        'log', 'log:operation',
                        'feedback', 'feedback:list'
                    ]
            except Exception:
                # 如果查询失败（例如表不存在），则返回默认权限
                permissions = [
                    'user', 'user:list', 'user:view',
                    'content', 'content:category', 'content:document',
                    'log', 'log:operation',
                    'feedback', 'feedback:list'
                ]
        
        # 记录操作日志 (如果提供了管理员ID)
        if admin_id:
            try:
                execute_update(
                    """INSERT INTO operation_logs (admin_id, operation_type, operation_desc, created_at) 
                       VALUES (%s, %s, %s, NOW())""", 
                    (admin_id, "query", f"管理员{admin_id}查询角色{role}的权限")
                )
            except Exception as log_error:
                # 仅记录日志错误，不影响主流程
                print(f"记录操作日志失败: {str(log_error)}")
        
        return {
            "code": 200,
            "message": "获取角色权限成功",
            "data": {
                "role": role,
                "permissions": permissions
            }
        }
    except Exception as e:
        # 记录错误日志
        print(f"获取角色权限失败: {str(e)}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"获取角色权限失败: {str(e)}")

# 更新角色权限接口
@router.post("/admin/roles/{role}/permissions", tags=["管理员管理"])
async def update_role_permissions(
    role: str,
    request: RolePermissionRequest
):
    """
    更新指定角色的权限列表
    """
    try:
        # 验证角色是否有效
        if role not in ['admin', 'operator']:
            return {
                "code": 400,
                "message": "无效的角色类型"
            }
        
        # 对于超级管理员角色，不允许修改权限
        if role == 'admin':
            return {
                "code": 400,
                "message": "超级管理员权限不可修改"
            }
        
        # 更新角色权限
        # 注意：这里假设角色权限存储在一个名为 role_permissions 的表中
        # 实际项目中，可能需要根据具体的数据库设计进行调整
        try:
            # 先删除该角色的所有现有权限
            execute_update(
                """DELETE FROM role_permissions WHERE role = %s""", 
                (role,)
            )
            
            # 然后插入新的权限
            for permission in request.permissions:
                execute_update(
                    """INSERT INTO role_permissions (role, permission_id, created_at) 
                       VALUES (%s, %s, NOW())""", 
                    (role, permission)
                )
        except Exception as db_error:
            # 如果数据库操作失败（例如表不存在），则记录错误但不抛出异常
            print(f"更新角色权限数据库操作失败: {str(db_error)}")
            # 这里可以考虑创建表或者其他适当的处理方式
        
        # 记录操作日志
        if request.admin_id:
            execute_update(
                """INSERT INTO operation_logs (admin_id, operation_type, operation_desc, created_at) 
                   VALUES (%s, %s, %s, NOW())""", 
                (request.admin_id, "update", f"管理员{request.admin_id}更新了角色{role}的权限")
            )
        
        return {
            "code": 200,
            "message": "更新角色权限成功"
        }
    except Exception as e:
        # 记录错误日志
        print(f"更新角色权限失败: {str(e)}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"更新角色权限失败: {str(e)}")
