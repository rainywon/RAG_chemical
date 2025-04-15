# 引入 FastAPI 中的 APIRouter 和 HTTPException 模块，用于创建路由和处理异常
from fastapi import APIRouter, HTTPException, Path, Query, Request, Depends
# 引入 Pydantic 中的 BaseModel 类，用于定义请求体的数据结构和验证
from pydantic import BaseModel
# 从数据库模块导入 execute_query 和 execute_update 函数，用于执行查询和更新操作
from database import execute_query, execute_update
# 引入类型提示
from typing import List, Dict, Any, Optional
# 引入日志模块记录系统日志
import logging
import traceback

# 初始化 APIRouter 实例，用于定义路由
router = APIRouter()

# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 定义系统配置更新模型
class SystemConfigUpdate(BaseModel):
    config_value: str
    description: Optional[str] = None

# 定义系统版本模型
class SystemVersionCreate(BaseModel):
    version_number: str
    knowledge_base_version: Optional[str] = None
    release_date: str
    update_notes: Optional[str] = None
    is_current: bool = False

# 设置当前版本请求模型
class SetCurrentVersionRequest(BaseModel):
    pass

# 获取当前管理员ID
async def get_current_admin(request: Request):
    try:
        # 从Authorization头获取token
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            raise HTTPException(status_code=401, detail="无效的认证信息")
        
        token = auth_header.split(' ')[1]
        
        # 查询admin_tokens表
        admin_result = execute_query(
            """SELECT admin_id FROM admin_tokens WHERE token = %s AND is_valid = 1 AND expire_at > NOW()""",
            (token,)
        )
        
        if not admin_result:
            raise HTTPException(status_code=401, detail="无效的token或token已过期")
        
        admin_id = admin_result[0]['admin_id']
        
        # 验证管理员是否存在
        admin_info = execute_query(
            """SELECT admin_id, full_name FROM admins WHERE admin_id = %s""",
            (admin_id,)
        )
        
        if not admin_info:
            raise HTTPException(status_code=401, detail="管理员不存在")
        
        return admin_id
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"管理员认证失败: {str(e)}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        raise HTTPException(status_code=401, detail=f"管理员认证失败: {str(e)}")

# 安全记录管理员操作
def log_admin_operation(admin_id: int, operation_type: str, description: str):
    if not admin_id:
        return
    
    try:
        # 获取管理员姓名
        admin_result = execute_query(
            """SELECT full_name FROM admins WHERE admin_id = %s""",
            (admin_id,)
        )
        
        admin_name = admin_result[0]['full_name'] if admin_result else f"管理员{admin_id}"
        
        # 在描述前添加管理员姓名
        full_description = f"{admin_name}{description}"
        
        execute_update(
            """INSERT INTO operation_logs (admin_id, operation_type, operation_desc, created_at) 
               VALUES (%s, %s, %s, NOW())""",
            (admin_id, operation_type, full_description)
        )
    except Exception as e:
        # 记录错误但不中断主要流程
        logger.error(f"记录操作日志失败: {str(e)}")

# 获取所有系统配置
@router.get("/admin/settings/system-configs", tags=["系统设置"])
async def get_system_configs(request: Request):
    """
    获取所有系统配置参数
    """
    try:
        # 获取管理员ID
        admin_id = await get_current_admin(request)
        
        configs = execute_query("SELECT * FROM system_configs ORDER BY config_id")
        
        # 记录操作日志
        log_admin_operation(admin_id, "查询", "查询系统配置参数")
        
        return {
            "code": 200,
            "message": "获取系统配置成功",
            "data": configs
        }
    except Exception as e:
        logger.error(f"获取系统配置失败: {str(e)}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"获取系统配置失败: {str(e)}")

# 更新系统配置
@router.put("/admin/settings/system-configs/{config_id}", tags=["系统设置"])
async def update_system_config(
    request: Request,
    config_id: int = Path(..., title="配置ID"),
    config_update: SystemConfigUpdate = None
):
    """
    更新指定ID的系统配置
    """
    try:
        # 获取管理员ID
        admin_id = await get_current_admin(request)
        
        # 检查配置是否存在
        existing_config = execute_query(
            "SELECT * FROM system_configs WHERE config_id = %s",
            (config_id,)
        )
        
        if not existing_config:
            raise HTTPException(status_code=404, detail="配置不存在")
        
        # 更新配置
        update_values = []
        update_fields = []
        
        if config_update.config_value is not None:
            update_fields.append("config_value = %s")
            update_values.append(config_update.config_value)
        
        if config_update.description is not None:
            update_fields.append("description = %s")
            update_values.append(config_update.description)
        
        if not update_fields:
            raise HTTPException(status_code=400, detail="没有要更新的字段")
        
        update_values.append(config_id)
        
        execute_update(
            f"UPDATE system_configs SET {', '.join(update_fields)}, updated_at = NOW() WHERE config_id = %s",
            tuple(update_values)
        )
        
        # 记录操作日志
        log_admin_operation(admin_id, "更新", f"更新系统配置{config_id}")
        
        # 获取更新后的配置
        updated_config = execute_query(
            "SELECT * FROM system_configs WHERE config_id = %s",
            (config_id,)
        )
        
        return {
            "code": 200,
            "message": "更新系统配置成功",
            "data": updated_config[0] if updated_config else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新系统配置失败: {str(e)}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"更新系统配置失败: {str(e)}")

# 获取所有系统版本
@router.get("/admin/settings/system-versions", tags=["系统设置"])
async def get_system_versions(request: Request):
    """
    获取所有系统版本信息
    """
    try:
        # 获取管理员ID
        admin_id = await get_current_admin(request)
        
        versions = execute_query(
            "SELECT * FROM system_versions ORDER BY version_id DESC"
        )
        
        # 记录操作日志
        log_admin_operation(admin_id, "查询", "查询系统版本列表")
        
        return {
            "code": 200,
            "message": "获取系统版本列表成功",
            "data": versions
        }
    except Exception as e:
        logger.error(f"获取系统版本失败: {str(e)}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"获取系统版本失败: {str(e)}")

# 添加新系统版本
@router.post("/admin/settings/system-versions", tags=["系统设置"])
async def create_system_version(
    request: Request,
    version: SystemVersionCreate
):
    """
    添加新的系统版本
    """
    try:
        # 获取管理员ID
        admin_id = await get_current_admin(request)
        
        # 如果设置为当前版本，则将其他版本设置为非当前版本
        if version.is_current:
            execute_update(
                "UPDATE system_versions SET is_current = 0 WHERE is_current = 1"
            )
        
        # 插入新版本
        version_id = execute_update(
            """
            INSERT INTO system_versions 
            (version_number, knowledge_base_version, release_date, update_notes, is_current) 
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                version.version_number,
                version.knowledge_base_version,
                version.release_date,
                version.update_notes,
                1 if version.is_current else 0
            ),
            return_last_id=True
        )
        
        # 记录操作日志
        log_admin_operation(admin_id, "创建", f"添加系统版本{version.version_number}")
        
        # 获取新添加的版本
        new_version = execute_query(
            "SELECT * FROM system_versions WHERE version_id = %s",
            (version_id,)
        )
        
        return {
            "code": 200,
            "message": "添加系统版本成功",
            "data": new_version[0] if new_version else None
        }
    except Exception as e:
        logger.error(f"添加系统版本失败: {str(e)}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"添加系统版本失败: {str(e)}")

# 设置当前版本
@router.put("/admin/settings/system-versions/{version_id}/set-current", tags=["系统设置"])
async def set_current_version(
    request: Request,
    version_id: int = Path(..., title="版本ID")
):
    """
    设置指定版本为当前版本
    """
    try:
        # 获取管理员ID
        admin_id = await get_current_admin(request)
        
        # 检查版本是否存在
        existing_version = execute_query(
            "SELECT * FROM system_versions WHERE version_id = %s",
            (version_id,)
        )
        
        if not existing_version or len(existing_version) == 0:
            raise HTTPException(status_code=404, detail="版本不存在")
        
        # 将其他所有版本设置为非当前版本
        execute_update(
            "UPDATE system_versions SET is_current = 0 WHERE is_current = 1"
        )
        
        # 将指定版本设置为当前版本
        execute_update(
            "UPDATE system_versions SET is_current = 1 WHERE version_id = %s",
            (version_id,)
        )
        
        # 记录操作日志
        log_admin_operation(admin_id, "更新", f"将版本{existing_version[0]['version_number']}设为当前版本")
        
        # 获取更新后的版本
        updated_version = execute_query(
            "SELECT * FROM system_versions WHERE version_id = %s",
            (version_id,)
        )
        
        return {
            "code": 200,
            "message": "设置当前版本成功",
            "data": updated_version[0] if updated_version else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"设置当前版本失败: {str(e)}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"设置当前版本失败: {str(e)}")