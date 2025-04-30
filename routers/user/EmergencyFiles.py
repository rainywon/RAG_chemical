from fastapi import APIRouter, HTTPException, Request, Query, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional
import os
from datetime import datetime
from database import execute_query, execute_update
from config import Config
import logging

router = APIRouter()
security = HTTPBearer()

# 获取配置实例
config = Config()
logger = logging.getLogger(__name__)

# 定义文件信息模型
class FileInfo(BaseModel):
    id: int
    name: str
    type: str
    size: int
    created_at: datetime
    updated_at: datetime

# 定义文件列表响应模型
class FileListResponse(BaseModel):
    code: int
    message: str
    data: List[FileInfo]
    total: int
    current_page: int
    total_pages: int

# 获取当前用户信息
async def get_current_user(request: Request):
    try:
        # 从Authorization头获取token
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            logger.error(f"无效的认证头: {auth_header}")
            raise HTTPException(status_code=401, detail="无效的认证信息")
        
        token = auth_header.split(' ')[1]
        logger.info(f"开始验证token: {token[:10]}...")
        
        # 查询admin_tokens表
        admin_result = execute_query(
            "SELECT admin_id, expire_at, is_valid FROM admin_tokens WHERE token = %s",
            (token,)
        )
        
        if admin_result:
            admin_token = admin_result[0]
            # 检查token是否有效
            if not admin_token['is_valid']:
                logger.error(f"Admin token已失效: {token[:10]}...")
                raise HTTPException(status_code=401, detail="token已失效")
                
            # 检查token是否过期
            if admin_token['expire_at'] < datetime.now():
                logger.error(f"Admin token已过期: {token[:10]}...")
                raise HTTPException(status_code=401, detail="token已过期")
                
            admin_id = admin_token['admin_id']
            # 获取管理员的详细信息
            admin_info = execute_query(
                "SELECT admin_id, full_name, phone_number FROM admins WHERE admin_id = %s",
                (admin_id,)
            )
            if admin_info:
                logger.info(f"管理员认证成功: {admin_info[0]['full_name']}")
                return {
                    "id": admin_id,
                    "role": "admin",
                    "name": admin_info[0]['full_name'],
                    "mobile": admin_info[0]['phone_number']
                }
            return {
                "id": admin_id,
                "role": "admin"
            }
        
        # 查询user_tokens表
        user_result = execute_query(
            "SELECT user_id, expire_at, is_valid FROM user_tokens WHERE token = %s",
            (token,)
        )
        
        if user_result:
            user_token = user_result[0]
            # 检查token是否有效
            if not user_token['is_valid']:
                logger.error(f"User token已失效: {token[:10]}...")
                raise HTTPException(status_code=401, detail="token已失效")
                
            # 检查token是否过期
            if user_token['expire_at'] < datetime.now():
                logger.error(f"User token已过期: {token[:10]}...")
                raise HTTPException(status_code=401, detail="token已过期")
            
            user_id = user_token['user_id']
            # 获取用户的详细信息
            user_info = execute_query(
                "SELECT user_id, mobile FROM users WHERE user_id = %s",
                (user_id,)
            )
            if user_info:
                logger.info(f"用户认证成功: {user_info[0]['mobile']}")
                return {
                    "id": user_id,
                    "role": "user",
                    "mobile": user_info[0]['mobile'],
                    "name": f"用户{user_info[0]['mobile']}"
                }
            return {
                "id": user_id,
                "role": "user"
            }
        
        # 如果没有找到对应token，抛出异常
        logger.error(f"未找到匹配的token: {token[:10]}...")
        raise HTTPException(status_code=401, detail="无效的token")
    except Exception as e:
        logger.error(f"验证用户身份时出错: {str(e)}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=401, detail=str(e))

# 记录操作日志
async def log_operation(user_info: dict, operation_type: str, operation_desc: str, request: Request = None):
    try:
        # 获取IP地址和用户代理
        ip_address = request.client.host if request else None
        user_agent = request.headers.get('user-agent') if request else None
        
        # 获取用户名
        user_name = ""
        if "name" in user_info:
            user_name = user_info["name"]
        elif "mobile" in user_info:
            user_name = f"用户{user_info['mobile']}"
        else:
            user_name = f"{'管理员' if user_info['role'] == 'admin' else '用户'}{user_info['id']}"
        
        # 创建操作日志记录
        if user_info['role'] == 'admin':
            execute_update(
                """INSERT INTO operation_logs (admin_id, operation_type, operation_desc, ip_address, user_agent, created_at) 
                   VALUES (%s, %s, %s, %s, %s, NOW())""",
                (user_info['id'], operation_type, f"{user_name}{operation_desc}", ip_address, user_agent)
            )
        else:
            execute_update(
                """INSERT INTO operation_logs (user_id, operation_type, operation_desc, ip_address, user_agent, created_at) 
                   VALUES (%s, %s, %s, %s, %s, NOW())""",
                (user_info['id'], operation_type, f"{user_name}{operation_desc}", ip_address, user_agent)
            )
    except Exception as e:
        # 记录错误但不中断主要流程
        logger.error(f"记录操作日志失败: {str(e)}")

# 获取文件列表
@router.get("/emergency_files/", response_model=FileListResponse)
async def get_emergency_files(
    request: Request,
    page: int = Query(1, description="页码"),
    page_size: int = Query(10, description="每页数量"),
    search: Optional[str] = Query(None, description="搜索关键词"),
    current_user: dict = Depends(get_current_user)
):
    try:
        # 记录操作日志
        await log_operation(
            current_user, 
            "查询", 
            f"查询应急资料库文件列表，页码：{page}，搜索关键词：{search if search else '无'}", 
            request
        )
        
        # 使用配置中的文件存储路径
        base_path = config.emergency_plan_path
        
        # 获取所有文件
        all_files = []
        for root, dirs, files in os.walk(base_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_info = {
                    'id': len(all_files) + 1,
                    'name': file,
                    'type': os.path.splitext(file)[1][1:].lower(),
                    'size': os.path.getsize(file_path),
                    'created_at': datetime.fromtimestamp(os.path.getctime(file_path)),
                    'updated_at': datetime.fromtimestamp(os.path.getmtime(file_path))
                }
                all_files.append(file_info)
        
        # 根据搜索条件过滤文件
        if search:
            search = search.lower()
            all_files = [f for f in all_files if search in f['name'].lower()]
        
        # 计算分页
        total = len(all_files)
        total_pages = (total + page_size - 1) // page_size
        start = (page - 1) * page_size
        end = start + page_size
        paginated_files = all_files[start:end]
        
        return {
            "code": 200,
            "message": "获取文件列表成功",
            "data": paginated_files,
            "total": total,
            "current_page": page,
            "total_pages": total_pages
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 下载文件
@router.get("/emergency_files/download/{file_id}")
async def download_emergency_file(
    request: Request,
    file_id: int,
    current_user: dict = Depends(get_current_user)
):
    try:
        # 使用配置中的文件存储路径
        base_path = config.safety_document_path
        
        # 获取所有文件
        all_files = []
        for root, dirs, files in os.walk(base_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_info = {
                    'id': len(all_files) + 1,
                    'path': file_path,
                    'name': file
                }
                all_files.append(file_info)
        
        # 查找指定ID的文件
        target_file = next((f for f in all_files if f['id'] == file_id), None)
        if not target_file:
            raise HTTPException(status_code=404, detail="文件不存在")
        
        # 记录操作日志
        await log_operation(
            current_user, 
            "下载文件", 
            f"下载了文件[{target_file['name']}]", 
            request
        )
        
        # 返回文件
        from fastapi.responses import FileResponse
        return FileResponse(
            target_file['path'],
            filename=target_file['name'],
            media_type='application/octet-stream'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 