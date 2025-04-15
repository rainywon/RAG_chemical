from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional, Tuple
import os
from datetime import datetime
from database import execute_query, execute_update
from config import Config
import jwt
from jwt.exceptions import InvalidTokenError
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

# 获取当前用户ID和角色
async def get_current_user(request: Request) -> Tuple[int, str]:
    try:
        # 从请求头获取Authorization
        auth_header = request.headers.get('Authorization')
        
        if not auth_header or not auth_header.startswith('Bearer '):
            raise HTTPException(status_code=401, detail="未提供有效的认证信息")
        
        # 提取token
        token = auth_header.split(' ')[1]
        
        # 首先查询admin_tokens表
        admin_result = execute_query(
            """SELECT admin_id FROM admin_tokens WHERE token = %s AND is_valid = 1""",
            (token,)
        )
        
        if admin_result and len(admin_result) > 0:
            admin_id = admin_result[0]['admin_id']
            return admin_id, "admin"
        
        # 如果不是管理员token，查询user_tokens表
        user_result = execute_query(
            """SELECT user_id FROM user_tokens WHERE token = %s AND is_valid = 1""",
            (token,)
        )
        
        if user_result and len(user_result) > 0:
            user_id = user_result[0]['user_id']
            return user_id, "user"
        
        # 如果两个表都查不到，说明token无效
        raise HTTPException(status_code=401, detail="无效的token")
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))

# 记录操作日志
async def log_operation(user_id: int, user_role: str, operation_type: str, operation_desc: str, request: Request = None):
    try:
        # 获取IP地址和用户代理
        ip_address = request.client.host if request else None
        user_agent = request.headers.get('user-agent') if request else None
        
        # 根据用户角色获取用户信息
        if user_role == "admin":
            # 从admins表获取管理员信息
            admin_info = execute_query(
                """SELECT admin_id, full_name FROM admins WHERE admin_id = %s""",
                (user_id,)
            )
            if not admin_info:
                return  # 如果管理员不存在，跳过日志记录
            
            admin_id, user_name = admin_info[0]['admin_id'], admin_info[0]['full_name']
            
            # 插入管理员操作日志
            execute_update(
                """INSERT INTO operation_logs (admin_id, operation_type, operation_desc, ip_address, user_agent, created_at) 
                   VALUES (%s, %s, %s, %s, %s, NOW())""",
                (admin_id, operation_type, f"{user_name}{operation_desc}", ip_address, user_agent)
            )
        else:
            # 从users表获取用户信息
            user_info = execute_query(
                """SELECT user_id, mobile FROM users WHERE user_id = %s""",
                (user_id,)
            )
            if not user_info:
                return  # 如果用户不存在，跳过日志记录
            
            user_id, user_name = user_info[0]['user_id'], user_info[0]['mobile']
            
            # 插入用户操作日志
            execute_update(
                """INSERT INTO operation_logs (user_id, operation_type, operation_desc, ip_address, user_agent, created_at) 
                   VALUES (%s, %s, %s, %s, %s, NOW())""",
                (user_id, operation_type, f"用户{user_name}{operation_desc}", ip_address, user_agent)
            )
    except Exception as e:
        # 记录错误但不中断主要流程
        logger.error(f"记录操作日志失败: {str(e)}")

# 获取文件列表
@router.get("/safety_files/", response_model=FileListResponse)
async def get_safety_files(
    request: Request,
    page: int = Query(1, description="页码"),
    page_size: int = Query(10, description="每页数量"),
    search: Optional[str] = Query(None, description="搜索关键词")
):
    try:
        user_id, user_role = await get_current_user(request)
        
        # 使用配置中的文件存储路径
        base_path = config.safety_document_path
        if not os.path.exists(base_path):
            raise HTTPException(status_code=500, detail="文件存储路径不存在")
        
        # 获取所有文件
        all_files = []
        for root, dirs, files in os.walk(base_path):
            for file in files:
                try:
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
                except Exception as e:
                    logger.error(f"处理文件 {file} 时出错: {str(e)}")
                    continue
        
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
        
        # 记录操作日志（放在最后，确保主要功能完成）
        try:
            await log_operation(
                user_id,
                user_role,
                "查询", 
                f"查询安全资料库文件列表，页码：{page}，搜索关键词：{search if search else '无'}", 
                request
            )
        except Exception as e:
            logger.error(f"记录操作日志失败: {str(e)}")
        
        return {
            "code": 200,
            "message": "获取文件列表成功",
            "data": paginated_files,
            "total": total,
            "current_page": page,
            "total_pages": total_pages
        }
    except Exception as e:
        logger.error(f"获取文件列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 下载文件
@router.get("/safety_files/download/{file_id}")
async def download_file(
    request: Request,
    file_id: int
):
    try:
        user_id, user_role = await get_current_user(request)
        
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
            user_id,
            user_role,
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
        logger.error(f"下载文件失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
