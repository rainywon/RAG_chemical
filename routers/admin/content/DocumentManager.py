# 引入 FastAPI 中的 APIRouter 和 HTTPException 模块，用于创建路由和处理异常
from fastapi import APIRouter, HTTPException, Query, UploadFile, File
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

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化 APIRouter 实例，用于定义路由
router = APIRouter()

# 设置文件存储路径
SAFETY_DOCUMENT_PATH = r"C:\Users\coins\Desktop\chemical_rag\data\标准性文件"
# C:\Users\coins\Desktop\chemical_rag\data\标准性文件
# C:\wu\RAG\data\safey_document
# 确保路径存在
os.makedirs(SAFETY_DOCUMENT_PATH, exist_ok=True)

# 安全记录管理员操作
def log_admin_operation(admin_id: Optional[int], operation_type: str, description: str):
    if not admin_id:
        return
    
    try:
        # 验证管理员是否存在
        result = execute_query("SELECT admin_id FROM admins WHERE admin_id = %s", (admin_id,))
        if result:
            execute_update(
                "INSERT INTO operation_logs (admin_id, operation_type, operation_desc) VALUES (%s, %s, %s)",
                (admin_id, operation_type, description)
            )
        else:
            logger.warning(f"管理员ID {admin_id} 不存在，无法记录操作日志")
    except Exception as e:
        # 记录错误但不中断主要流程
        logger.error(f"记录操作日志失败: {str(e)}")

# 定义文件信息模型
class FileInfo(BaseModel):
    id: str
    fileName: str
    fileType: str
    fileSize: int
    createdTime: str
    lastModified: str
    path: str

# 定义批量删除请求模型
class BatchDeleteRequest(BaseModel):
    file_ids: List[str]
    admin_id: Optional[int] = None

# 获取文件列表
@router.get("/admin/content/safety-documents", tags=["安全资料库"])
def get_document_list(
    search_query: Optional[str] = Query(None, description="搜索关键词"),
    file_type: Optional[str] = Query(None, description="文件类型"),
    start_date: Optional[str] = Query(None, description="开始日期"),
    end_date: Optional[str] = Query(None, description="结束日期"),
    sort_by: Optional[str] = Query("name-asc", description="排序方式"),
    page: int = Query(1, description="页码"),
    page_size: int = Query(20, description="每页数量"),
    admin_id: Optional[int] = Query(None, description="管理员ID")
):
    """
    获取安全资料库文件列表，支持搜索、筛选和排序
    """
    try:
        # 记录请求参数用于调试
        
        # 记录操作日志（不阻止主要功能）
        log_admin_operation(admin_id, "查询", f"管理员{admin_id}查询安全资料库文件列表")
        
        # 获取所有文件
        file_list = []
        file_paths = glob.glob(os.path.join(SAFETY_DOCUMENT_PATH, "*.*"))
        
        # 记录查找到的总文件数
        
        for i, file_path in enumerate(file_paths):
            file_stats = os.stat(file_path)
            file_path_obj = Path(file_path)
            file_name = file_path_obj.name
            file_ext = file_path_obj.suffix.lower()  # 转为小写便于比较
            
            # 只处理 pdf、doc、docx 文件
            if file_ext not in ['.pdf', '.doc', '.docx']:
                continue
            
            file_info = {
                "id": str(i + 1),
                "fileName": file_name,
                "fileType": file_ext,
                "fileSize": file_stats.st_size,
                "createdTime": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                "lastModified": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                "path": file_path
            }
            
            # 应用搜索过滤
            if search_query and search_query.lower() not in file_name.lower():
                continue
                
            # 应用文件类型过滤
            if file_type and file_ext != file_type:
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
        
        # 记录过滤后的文件数
        
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
        
        # 记录分页信息
        
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
@router.post("/admin/content/safety-documents/upload", tags=["安全资料库"])
def upload_documents(
    files: List[UploadFile] = File(...),
    admin_id: Optional[int] = Query(None, description="管理员ID")
):
    """
    上传文件到安全资料库
    """
    try:
        uploaded_files = []
        
        for file in files:
            # 获取文件类型（小写）
            file_extension = os.path.splitext(file.filename)[1].lower()
            
            # 检查文件类型
            if file_extension not in ['.pdf', '.doc', '.docx']:
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "message": f"不支持的文件类型: {file.filename}, 只支持PDF、Word文档(.pdf, .doc, .docx)"}
                )
            
            # 检查文件大小
            content = file.read()
            if len(content) > 20 * 1024 * 1024:  # 20MB
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "message": f"文件过大: {file.filename}, 最大允许20MB"}
                )
            
            # 将文件指针重置到开始位置
            file.seek(0)
            
            # 保存文件
            file_path = os.path.join(SAFETY_DOCUMENT_PATH, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            uploaded_files.append(file.filename)
        
        # 记录操作日志（不阻止主要功能）
        log_admin_operation(admin_id, "上传文件", f"管理员{admin_id}上传了{len(uploaded_files)}个安全资料文件")
        
        return {"success": True, "message": f"成功上传 {len(uploaded_files)} 个文件", "files": uploaded_files}
    except Exception as e:
        logger.error(f"文件上传失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")

# 下载文件
@router.get("/admin/content/safety-documents/download/{file_name}", tags=["安全资料库"])
def download_document(
    file_name: str,
    admin_id: Optional[int] = Query(None, description="管理员ID")
):
    """
    下载安全资料库文件
    """
    try:
        file_path = os.path.join(SAFETY_DOCUMENT_PATH, file_name)
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="文件不存在")
        
        # 设置适当的Content-Type
        file_extension = os.path.splitext(file_name)[1].lower()
        media_type = 'application/octet-stream'
        if file_extension == '.pdf':
            media_type = 'application/pdf'
        elif file_extension == '.doc':
            media_type = 'application/msword'
        elif file_extension == '.docx':
            media_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        
        # 记录操作日志（不阻止主要功能）
        log_admin_operation(admin_id, "下载文件", f"管理员{admin_id}下载了安全资料文件{file_name}")
        
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

# 删除单个文件
@router.delete("/admin/content/safety-documents/{file_name}", tags=["安全资料库"])
def delete_document(
    file_name: str,
    admin_id: Optional[int] = Query(None, description="管理员ID")
):
    """
    删除安全资料库中的单个文件
    """
    try:
        file_path = os.path.join(SAFETY_DOCUMENT_PATH, file_name)
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="文件不存在")
        
        # 删除文件
        os.remove(file_path)
        
        # 记录操作日志（不阻止主要功能）
        log_admin_operation(admin_id, "删除", f"管理员{admin_id}删除了安全资料文件{file_name}")
        
        return {"success": True, "message": f"成功删除文件: {file_name}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文件删除失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文件删除失败: {str(e)}")

# 批量删除文件
@router.post("/admin/content/safety-documents/batch-delete", tags=["安全资料库"])
def batch_delete_documents(
    request: BatchDeleteRequest
):
    """
    批量删除安全资料库文件
    """
    try:
        deleted_files = []
        failed_files = []
        file_paths = glob.glob(os.path.join(SAFETY_DOCUMENT_PATH, "*.*"))
        
        # 获取所有文件的ID和名称映射
        file_map = {}
        for i, file_path in enumerate(file_paths):
            file_name = os.path.basename(file_path)
            file_map[str(i + 1)] = {"name": file_name, "path": file_path}
        
        # 删除指定ID的文件
        for file_id in request.file_ids:
            if file_id in file_map:
                file_info = file_map[file_id]
                try:
                    os.remove(file_info["path"])
                    deleted_files.append(file_info["name"])
                except Exception as e:
                    failed_files.append({"name": file_info["name"], "error": str(e)})
        
        # 记录操作日志（不阻止主要功能）
        log_admin_operation(request.admin_id, "删除", f"管理员{request.admin_id}批量删除了{len(deleted_files)}个安全资料文件")
        
        return {
            "success": True,
            "message": f"成功删除 {len(deleted_files)} 个文件，失败 {len(failed_files)} 个",
            "deleted_files": deleted_files,
            "failed_files": failed_files
        }
    except Exception as e:
        logger.error(f"批量删除文件失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"批量删除文件失败: {str(e)}")

# 获取文件预览
@router.get("/admin/content/safety-documents/preview/{file_name}", tags=["安全资料库"])
def preview_document(
    file_name: str,
    admin_id: Optional[int] = Query(None, description="管理员ID")
):
    """
    获取安全资料文件的预览信息，不包含文件内容，只返回基本信息
    """
    try:
        file_path = os.path.join(SAFETY_DOCUMENT_PATH, file_name)
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="文件不存在")
        
        # 获取文件类型
        file_extension = os.path.splitext(file_name)[1].lower()
        
        # 检查文件类型
        if file_extension not in ['.pdf', '.doc', '.docx']:
            raise HTTPException(status_code=400, detail="不支持的文件类型，只支持PDF和Word文档")
        
        # 获取文件基本信息
        file_stats = os.stat(file_path)
        file_size = file_stats.st_size
        created_time = datetime.fromtimestamp(file_stats.st_ctime).isoformat()
        modified_time = datetime.fromtimestamp(file_stats.st_mtime).isoformat()
        
        # 记录操作日志（不阻止主要功能）
        log_admin_operation(admin_id, "查询", f"管理员{admin_id}预览了安全资料文件{file_name}")
        
        # 由于无法直接预览PDF或Word文档内容，返回其基本信息
        preview_info = {
            "fileName": file_name,
            "fileType": file_extension,
            "fileSize": file_size,
            "createdTime": created_time,
            "lastModified": modified_time,
            "previewAvailable": False,  # 标记实际内容预览不可用
            "message": "PDF和Word文档内容无法直接预览，请下载后查看"
        }
        
        return {"success": True, "fileInfo": preview_info}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文件预览失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文件预览失败: {str(e)}")
