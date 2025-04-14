from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional
import os
from datetime import datetime
from database import execute_query, execute_update
from config import Config

router = APIRouter()
security = HTTPBearer()

# 获取配置实例
config = Config()

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

# 获取当前用户ID
async def get_current_user(request: Request):
    user_id = request.headers.get('X-User-ID')
    if not user_id:
        raise HTTPException(status_code=401, detail="未提供用户ID")
    return int(user_id)

# 记录操作日志
async def log_operation(user_id: int, operation_type: str, operation_desc: str, request: Request = None):
    try:
        # 获取IP地址和用户代理
        ip_address = request.client.host if request else None
        user_agent = request.headers.get('user-agent') if request else None
        
        # 创建操作日志记录
        execute_update(
            """INSERT INTO operation_logs (user_id, operation_type, operation_desc, ip_address, user_agent, created_at) 
               VALUES (%s, %s, %s, %s, %s, NOW())""",
            (user_id, operation_type, operation_desc, ip_address, user_agent)
        )
    except Exception as e:
        # 记录错误但不中断主要流程
        print(f"记录操作日志失败: {str(e)}")

# 获取文件列表
@router.get("/safety_files/", response_model=FileListResponse)
async def get_safety_files(
    request: Request,
    page: int = Query(1, description="页码"),
    page_size: int = Query(10, description="每页数量"),
    search: Optional[str] = Query(None, description="搜索关键词")
):
    try:
        user_id = await get_current_user(request)
        
        # 记录操作日志
        await log_operation(
            user_id, 
            "查询", 
            f"用户{user_id}查询安全资料库文件列表，页码：{page}，搜索关键词：{search if search else '无'}", 
            request
        )
        
        # 使用配置中的文件存储路径
        base_path = config.safety_files_path
        
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
@router.get("/safety_files/download/{file_id}")
async def download_file(
    request: Request,
    file_id: int
):
    try:
        user_id = await get_current_user(request)
        
        # 使用配置中的文件存储路径
        base_path = config.safety_files_path
        
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
            "下载文件", 
            f"用户{user_id}下载了文件[{target_file['name']}]", 
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
