# 引入 FastAPI 中的 APIRouter 和 HTTPException 模块，用于创建路由和处理异常
from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Request, Depends
from fastapi.responses import FileResponse, JSONResponse
# 引入 Pydantic 中的 BaseModel 类，用于定义请求体的数据结构和验证
from pydantic import BaseModel
# 从数据库模块导入 execute_query 和 execute_update 函数，用于执行查询和更新操作
from database import execute_query, execute_update
# 引入时间模块处理日期范围
from datetime import datetime, timedelta
# 引入 typing 模块中的 Optional 和 List 类型
from typing import Optional, List, Dict, Any
# 引入操作系统相关模块
import os
import shutil
from pathlib import Path
import glob
import logging
from config import Config
import traceback
# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化 APIRouter 实例，用于定义路由
router = APIRouter()
config=Config()
# 设置文件存储路径
EMERGENCY_PLAN_PATH = config.emergency_plan_path
# C:\Users\coins\Desktop\chemical_rag\data\标准性文件
# C:\wu\RAG\data\safey_document
# 确保路径存在
os.makedirs(EMERGENCY_PLAN_PATH, exist_ok=True)

# 用于批量删除文件的请求模型
class BatchDeleteRequest(BaseModel):
    file_ids: List[str]

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
            raise HTTPException(status_code=401, detail="无效的管理员令牌或令牌已过期")
        
        admin_id = admin_result[0]['admin_id']
        
        # 验证管理员是否存在
        admin = execute_query("""SELECT * FROM admins WHERE admin_id = %s AND status = 1""", (admin_id,))
        if not admin:
            raise HTTPException(status_code=401, detail="管理员不存在或已被禁用")
            
        return admin_id
    except Exception as e:
        logger.error(f"管理员认证失败: {str(e)}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        raise HTTPException(status_code=401, detail=f"管理员认证失败: {str(e)}")

# 记录管理员操作日志
def log_admin_operation(admin_id, operation_type, description):
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
            """INSERT INTO operation_logs 
               (admin_id, operation_type, operation_desc, created_at) 
               VALUES (%s, %s, %s, NOW())""",
            (admin_id, operation_type, full_description)
        )
    except Exception as e:
        logger.error(f"记录操作日志失败: {str(e)}")
        # 操作日志记录失败不会阻止主要功能

# 获取文件列表
@router.get("/admin/content/emergency-plans", tags=["事故案例管理"])
async def get_file_list(
    request: Request,
    search_query: Optional[str] = Query(None, description="搜索关键词"),
    file_type: Optional[str] = Query(None, description="文件类型"),
    start_date: Optional[str] = Query(None, description="开始日期"),
    end_date: Optional[str] = Query(None, description="结束日期"),
    sort_by: Optional[str] = Query("name-asc", description="排序方式"),
    page: int = Query(1, description="页码"),
    page_size: int = Query(20, description="每页数量")
):
    """
    获取事故案例文件列表，支持搜索、筛选和排序
    """
    try:
        # 获取管理员ID
        admin_id = await get_current_admin(request)
        
        # 记录操作日志（不阻止主要功能）
        log_admin_operation(admin_id, "查询", "查询事故案例文件列表")
        
        # 获取所有文件（包括子目录）
        file_list = []
        for root, dirs, files in os.walk(EMERGENCY_PLAN_PATH):
            for file in files:
                file_path = os.path.join(root, file)
                file_stats = os.stat(file_path)
                file_path_obj = Path(file_path)
                file_name = file_path_obj.name
                file_ext = file_path_obj.suffix
                
                # 只接受 PDF, DOC, DOCX 文件
                if file_ext.lower() not in ['.pdf', '.doc', '.docx']:
                    continue
                
                # 获取相对路径（相对于EMERGENCY_PLAN_PATH）
                rel_path = os.path.relpath(root, EMERGENCY_PLAN_PATH)
                category = rel_path if rel_path != '.' else '根目录'
                
                file_info = {
                    "id": str(len(file_list) + 1),
                    "fileName": file_name,
                    "fileType": file_ext,
                    "fileSize": file_stats.st_size,
                    "createdTime": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                    "lastModified": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                    "path": file_path,
                    "category": category
                }
                
                # 应用搜索过滤
                if search_query and search_query.lower() not in file_name.lower():
                    continue
                    
                # 应用文件类型过滤
                if file_type and file_ext.lower() != f".{file_type.lower()}":
                    continue
                    
                # 应用日期范围过滤
                if start_date or end_date:
                    file_date = datetime.fromtimestamp(file_stats.st_mtime)
                    
                    if start_date:
                        start_datetime = datetime.fromisoformat(start_date)
                        if file_date < start_datetime:
                            continue
                            
                    if end_date:
                        end_datetime = datetime.fromisoformat(end_date)
                        end_datetime = end_datetime.replace(hour=23, minute=59, second=59)
                        if file_date > end_datetime:
                            continue
                
                file_list.append(file_info)
        
        # 应用排序
        if sort_by:
            field, direction = sort_by.split('-')
            reverse = direction == 'desc'
            
            if field == 'name':
                file_list.sort(key=lambda x: x["fileName"], reverse=reverse)
            elif field == 'size':
                file_list.sort(key=lambda x: x["fileSize"], reverse=reverse)
            elif field == 'date':
                file_list.sort(key=lambda x: x["lastModified"], reverse=reverse)
            elif field == 'category':
                file_list.sort(key=lambda x: x["category"], reverse=reverse)
        
        # 计算总数
        total_count = len(file_list)
        
        # 确保页码合法
        if page < 1:
            page = 1
        
        # 确保每页数量合法
        if page_size < 1:
            page_size = 20
        elif page_size > 100:
            page_size = 100  # 限制最大每页数量为100
        
        # 分页
        start_idx = (page - 1) * page_size
        
        # 确保起始索引不超出范围
        if start_idx >= total_count:
            if total_count > 0:
                page = (total_count - 1) // page_size + 1
                start_idx = (page - 1) * page_size
            else:
                start_idx = 0
        
        end_idx = min(start_idx + page_size, total_count)
        paginated_files = file_list[start_idx:end_idx]
        
        return {
            "success": True,
            "data": {
                "files": paginated_files,
                "total": total_count,
                "page": page,
                "pageSize": page_size
            }
        }
    except Exception as e:
        logger.error(f"获取文件列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取文件列表失败: {str(e)}")

# 上传文件
@router.post("/admin/content/emergency-plans/upload", tags=["事故案例管理"])
async def upload_files(
    request: Request,
    files: List[UploadFile] = File(...)
):
    """
    上传文件到事故案例库
    """
    try:
        # 获取管理员ID
        admin_id = await get_current_admin(request)
        
        uploaded_files = []
        
        for file in files:
            # 检查文件类型
            file_ext = os.path.splitext(file.filename)[1].lower()
            if file_ext not in ['.pdf', '.doc', '.docx']:
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "message": f"不支持的文件类型: {file.filename}, 只支持 PDF, DOC, DOCX 文件"}
                )
            
            # 检查文件大小
            content = file.read()
            if len(content) > 50 * 1024 * 1024:  # 50MB
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "message": f"文件过大: {file.filename}, 最大允许50MB"}
                )
            
            # 将文件指针重置到开始位置
            file.seek(0)
            
            # 保存文件
            file_path = os.path.join(EMERGENCY_PLAN_PATH, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            uploaded_files.append(file.filename)
        
        # 记录操作日志（不阻止主要功能）
        log_admin_operation(admin_id, "上传文件", f"上传了{len(uploaded_files)}个事故案例文件")
        
        return {"success": True, "message": f"成功上传 {len(uploaded_files)} 个事故案例文件", "files": uploaded_files}
    except Exception as e:
        logger.error(f"文件上传失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")

# 下载文件
@router.get("/admin/content/emergency-plans/download/{file_name}", tags=["事故案例管理"])
async def download_file(
    request: Request,
    file_name: str
):
    """
    下载事故案例文件
    """
    try:
        # 获取管理员ID
        admin_id = await get_current_admin(request)
        
        file_path = os.path.join(EMERGENCY_PLAN_PATH, file_name)
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="文件不存在")
        
        # 记录操作日志（不阻止主要功能）
        log_admin_operation(admin_id, "下载文件", f"下载事故案例文件{file_name}")
        
        # 确定媒体类型
        file_ext = os.path.splitext(file_name)[1].lower()
        media_type = 'application/pdf' if file_ext == '.pdf' else 'application/octet-stream'
        
        return FileResponse(
            path=file_path, 
            filename=file_name,
            media_type=media_type
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文件下载失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文件下载失败: {str(e)}")

# 查看PDF文件
@router.get("/admin/content/emergency-plans/view/{file_name}", tags=["事故案例管理"])
async def view_file(
    request: Request,
    file_name: str,
    token: Optional[str] = Query(None, description="管理员Token")
):
    """
    查看PDF文件
    """
    try:
        # 获取管理员ID（从token查询参数或Authorization头）
        admin_id = None
        if token:
            # 从URL参数中获取token
            admin_result = execute_query(
                """SELECT admin_id FROM admin_tokens WHERE token = %s AND is_valid = 1 AND expire_at > NOW()""",
                (token,)
            )
            if admin_result:
                admin_id = admin_result[0]['admin_id']
        
        if admin_id is None:
            # 如果URL参数中没有有效token，则从Authorization头获取
            admin_id = await get_current_admin(request)
            
        file_path = os.path.join(EMERGENCY_PLAN_PATH, file_name)
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="文件不存在")
        
        # 检查文件是否为PDF
        if not file_name.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="只支持查看PDF文件")
        
        # 记录操作日志（不阻止主要功能）
        log_admin_operation(admin_id, "查询", f"在线查看了PDF文件{file_name}")
        
        return FileResponse(
            path=file_path, 
            filename=file_name,
            media_type='application/pdf'
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文件查看失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文件查看失败: {str(e)}")

# 删除单个文件
@router.delete("/admin/content/emergency-plans/{file_name}", tags=["事故案例管理"])
async def delete_file(
    request: Request,
    file_name: str
):
    """
    删除事故案例中的单个文件
    """
    try:
        # 获取管理员ID
        admin_id = await get_current_admin(request)
        
        file_path = os.path.join(EMERGENCY_PLAN_PATH, file_name)
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="文件不存在")
        
        # 删除文件
        os.remove(file_path)
        
        # 记录操作日志（不阻止主要功能）
        log_admin_operation(admin_id, "删除", f"删除了事故案例文件{file_name}")
        
        return {"success": True, "message": f"成功删除文件: {file_name}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文件删除失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文件删除失败: {str(e)}")

# 批量删除文件
@router.post("/admin/content/emergency-plans/batch-delete", tags=["事故案例管理"])
async def batch_delete_files(
    request: Request,
    batch_request: BatchDeleteRequest
):
    """
    批量删除事故案例文件
    """
    try:
        # 获取管理员ID
        admin_id = await get_current_admin(request)
        
        deleted_files = []
        failed_files = []
        file_paths = glob.glob(os.path.join(EMERGENCY_PLAN_PATH, "*.*"))
        
        # 获取所有文件的ID和名称映射
        file_map = {}
        for i, file_path in enumerate(file_paths):
            file_name = os.path.basename(file_path)
            file_map[str(i + 1)] = {"name": file_name, "path": file_path}
        
        # 删除指定ID的文件
        for file_id in batch_request.file_ids:
            if file_id in file_map:
                file_info = file_map[file_id]
                try:
                    os.remove(file_info["path"])
                    deleted_files.append(file_info["name"])
                except Exception as e:
                    failed_files.append({"name": file_info["name"], "error": str(e)})
        
        # 记录操作日志（不阻止主要功能）
        log_admin_operation(admin_id, "删除", f"批量删除了{len(deleted_files)}个事故案例文件")
        
        return {
            "success": True,
            "message": f"成功删除 {len(deleted_files)} 个文件，失败 {len(failed_files)} 个",
            "deleted_files": deleted_files,
            "failed_files": failed_files
        }
    except Exception as e:
        logger.error(f"批量删除文件失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"批量删除文件失败: {str(e)}")
