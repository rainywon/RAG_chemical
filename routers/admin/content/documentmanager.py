# 引入 FastAPI 中的 APIRouter 和 HTTPException 模块，用于创建路由和处理异常
from fastapi import APIRouter, HTTPException, Depends, Query, Form, UploadFile, File
# 引入 Pydantic 中的 BaseModel 类，用于定义请求体的数据结构和验证
from pydantic import BaseModel
# 从数据库模块导入 execute_query 和 execute_update 函数，用于执行查询和更新操作
from database import execute_query, execute_update
# 引入时间模块处理日期
from datetime import datetime, date
# 引入 typing 模块中的 Optional 和 List 类型
from typing import Optional, List
# 导入路径和文件处理相关
import os, uuid, shutil
from fastapi.responses import FileResponse
from pathlib import Path
# 引入管理员认证依赖函数
from routers.login import get_current_admin

# 初始化 APIRouter 实例，用于定义路由
router = APIRouter()

# 定义文档请求体模型
class DocumentRequest(BaseModel):
    title: str
    category_id: int
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    file_type: Optional[str] = None
    author: Optional[str] = None
    publish_date: Optional[str] = None
    description: Optional[str] = None
    is_published: Optional[int] = 1
    admin_id: Optional[int] = None

# 定义文档状态请求体模型
class DocumentStatusRequest(BaseModel):
    is_published: int
    admin_id: Optional[int] = None

# 文档上传路径
DOCUMENT_UPLOAD_DIR = Path("data/documents")
# 确保上传目录存在
DOCUMENT_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# 获取知识文档列表接口
@router.get("/admin/content/documents", tags=["内容管理"])
async def get_documents(
    page: int = Query(1, description="页码"),
    page_size: int = Query(10, description="每页数量"),
    title: Optional[str] = Query(None, description="标题筛选"),
    category_id: Optional[int] = Query(None, description="分类ID筛选"),
    is_published: Optional[int] = Query(None, description="发布状态筛选")
):
    """
    获取知识文档列表，支持分页和筛选
    """
    try:
        # 计算偏移量
        offset = (page - 1) * page_size
        
        # 构建基础查询SQL
        query = """
            SELECT 
                d.document_id, d.title, d.file_path, d.file_size, d.file_type, 
                d.author, d.publish_date, d.description, d.is_published, 
                d.view_count, d.download_count, d.created_at, d.updated_at,
                c.category_id, c.category_name
            FROM 
                knowledge_documents d
            LEFT JOIN 
                knowledge_categories c ON d.category_id = c.category_id
            WHERE 1=1
        """
        params = []
        
        # 添加筛选条件
        if title:
            query += " AND d.title LIKE %s"
            params.append(f"%{title}%")
        
        if category_id:
            query += " AND d.category_id = %s"
            params.append(category_id)
        
        if is_published is not None:
            query += " AND d.is_published = %s"
            params.append(is_published)
        
        # 查询符合条件的总记录数
        count_query = f"SELECT COUNT(*) as count FROM ({query}) as filtered_docs"
        count_result = execute_query(count_query, tuple(params))
        total_count = count_result[0]['count'] if count_result else 0
        
        # 添加排序和分页
        query += " ORDER BY d.updated_at DESC LIMIT %s OFFSET %s"
        params.append(page_size)
        params.append(offset)
        
        # 查询文档列表
        document_list = execute_query(query, tuple(params))
        
        # 处理日期时间格式
        for doc in document_list:
            doc['publish_date'] = doc['publish_date'].strftime("%Y-%m-%d") if doc['publish_date'] else None
            doc['created_at'] = doc['created_at'].strftime("%Y-%m-%d %H:%M:%S") if doc['created_at'] else None
            doc['updated_at'] = doc['updated_at'].strftime("%Y-%m-%d %H:%M:%S") if doc['updated_at'] else None
        
        return {
            "code": 200,
            "message": "获取文档列表成功",
            "data": {
                "documents": document_list,
                "total": total_count
            }
        }
    except Exception as e:
        # 记录错误日志
        print(f"获取文档列表失败: {str(e)}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {str(e)}")

# 文件上传接口
@router.post("/admin/content/documents/upload", tags=["内容管理"])
async def upload_document_file(file: UploadFile = File(...)):
    """
    上传文档文件
    """
    try:
        # 检查文件类型
        allowed_types = ["application/pdf", "application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
        content_type = file.content_type
        
        if content_type not in allowed_types:
            return {
                "code": 400,
                "message": "不支持的文件类型，仅支持PDF、DOC和DOCX文件"
            }
        
        # 生成唯一文件名
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = DOCUMENT_UPLOAD_DIR / unique_filename
        
        # 保存文件
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 获取文件大小（KB）
        file_size = os.path.getsize(file_path) // 1024
        
        # 获取文件类型
        file_type = file_extension.lstrip('.').upper()
        
        return {
            "code": 200,
            "message": "文件上传成功",
            "data": {
                "file_path": str(unique_filename),
                "file_size": file_size,
                "file_type": file_type
            }
        }
    except Exception as e:
        # 记录错误日志
        print(f"文件上传失败: {str(e)}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")

# 创建知识文档接口
@router.post("/admin/content/documents", tags=["内容管理"])
async def create_document(request: DocumentRequest):
    """
    创建新的知识文档
    """
    try:
        # 检查分类是否存在
        category_check = execute_query(
            """SELECT * FROM knowledge_categories WHERE category_id = %s""", 
            (request.category_id,)
        )
        
        if not category_check:
            return {
                "code": 404,
                "message": "所选分类不存在"
            }
        
        # 检查文件路径是否存在
        if request.file_path:
            file_check = os.path.exists(DOCUMENT_UPLOAD_DIR / request.file_path)
            if not file_check:
                return {
                    "code": 404,
                    "message": "上传的文件不存在"
                }
        
        # 转换发布日期格式
        publish_date = None
        if request.publish_date:
            try:
                publish_date = datetime.strptime(request.publish_date, "%Y-%m-%d").date()
            except ValueError:
                return {
                    "code": 400,
                    "message": "发布日期格式错误，应为YYYY-MM-DD"
                }
        
        # 插入文档记录
        insert_result = execute_update(
            """
            INSERT INTO knowledge_documents 
                (category_id, title, file_path, file_size, file_type, author, 
                 publish_date, description, is_published) 
            VALUES 
                (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, 
            (
                request.category_id, request.title, request.file_path, 
                request.file_size, request.file_type, request.author,
                publish_date, request.description, request.is_published
            )
        )
        
        if not insert_result or 'lastrowid' not in insert_result:
            return {
                "code": 500,
                "message": "创建文档失败"
            }
        
        new_document_id = insert_result['lastrowid']
        
        # 记录操作日志
        if request.admin_id:
            try:
                execute_update(
                    """
                    INSERT INTO operation_logs 
                        (admin_id, operation_type, operation_desc, created_at) 
                    VALUES 
                        (%s, %s, %s, NOW())
                    """, 
                    (request.admin_id, "创建文档", f"管理员上传了新文档'{request.title}'")
                )
            except Exception as log_error:
                print(f"记录操作日志失败: {str(log_error)}")
        
        return {
            "code": 200,
            "message": "创建文档成功",
            "data": {
                "document_id": new_document_id
            }
        }
    except Exception as e:
        # 记录错误日志
        print(f"创建文档失败: {str(e)}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"创建文档失败: {str(e)}")

# 更新文档接口
@router.put("/admin/content/documents/{document_id}", tags=["内容管理"])
async def update_document(document_id: int, request: DocumentRequest):
    """
    更新现有知识文档
    """
    try:
        # 检查文档是否存在
        document_check = execute_query(
            """SELECT * FROM knowledge_documents WHERE document_id = %s""", 
            (document_id,)
        )
        
        if not document_check:
            return {
                "code": 404,
                "message": "文档不存在"
            }
        
        # 检查分类是否存在
        category_check = execute_query(
            """SELECT * FROM knowledge_categories WHERE category_id = %s""", 
            (request.category_id,)
        )
        
        if not category_check:
            return {
                "code": 404,
                "message": "所选分类不存在"
            }
        
        # 转换发布日期格式
        publish_date = None
        if request.publish_date:
            try:
                publish_date = datetime.strptime(request.publish_date, "%Y-%m-%d").date()
            except ValueError:
                return {
                    "code": 400,
                    "message": "发布日期格式错误，应为YYYY-MM-DD"
                }
        
        # 更新文档记录
        execute_update(
            """
            UPDATE knowledge_documents 
            SET 
                category_id = %s, 
                title = %s, 
                author = %s, 
                publish_date = %s, 
                description = %s, 
                is_published = %s,
                updated_at = NOW()
            WHERE 
                document_id = %s
            """, 
            (
                request.category_id, request.title, request.author,
                publish_date, request.description, request.is_published,
                document_id
            )
        )
        
        # 记录操作日志
        if request.admin_id:
            try:
                execute_update(
                    """
                    INSERT INTO operation_logs 
                        (admin_id, operation_type, operation_desc, created_at) 
                    VALUES 
                        (%s, %s, %s, NOW())
                    """, 
                    (request.admin_id, "更新文档", f"管理员更新了文档'{request.title}'(ID:{document_id})")
                )
            except Exception as log_error:
                print(f"记录操作日志失败: {str(log_error)}")
        
        return {
            "code": 200,
            "message": "更新文档成功"
        }
    except Exception as e:
        # 记录错误日志
        print(f"更新文档失败: {str(e)}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"更新文档失败: {str(e)}")

# 更新文档发布状态接口
@router.put("/admin/content/documents/{document_id}/status", tags=["内容管理"])
async def update_document_status(document_id: int, request: DocumentStatusRequest):
    """
    更新文档的发布状态
    """
    try:
        # 检查文档是否存在
        document_check = execute_query(
            """SELECT * FROM knowledge_documents WHERE document_id = %s""", 
            (document_id,)
        )
        
        if not document_check:
            return {
                "code": 404,
                "message": "文档不存在"
            }
        
        document_title = document_check[0]['title']
        old_status = document_check[0]['is_published']
        new_status = request.is_published
        
        # 如果状态没有变化，直接返回成功
        if old_status == new_status:
            return {
                "code": 200,
                "message": "文档状态未变化"
            }
        
        # 更新文档状态
        execute_update(
            """
            UPDATE knowledge_documents 
            SET 
                is_published = %s,
                updated_at = NOW()
            WHERE 
                document_id = %s
            """, 
            (new_status, document_id)
        )
        
        # 记录操作日志
        if request.admin_id:
            action = "发布" if new_status == 1 else "取消发布"
            try:
                execute_update(
                    """
                    INSERT INTO operation_logs 
                        (admin_id, operation_type, operation_desc, created_at) 
                    VALUES 
                        (%s, %s, %s, NOW())
                    """, 
                    (request.admin_id, f"{action}文档", f"管理员{action}了文档'{document_title}'(ID:{document_id})")
                )
            except Exception as log_error:
                print(f"记录操作日志失败: {str(log_error)}")
        
        return {
            "code": 200,
            "message": "更新文档状态成功"
        }
    except Exception as e:
        # 记录错误日志
        print(f"更新文档状态失败: {str(e)}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"更新文档状态失败: {str(e)}")

# 删除文档接口
@router.delete("/admin/content/documents/{document_id}", tags=["内容管理"])
async def delete_document(document_id: int, admin_id: Optional[int] = None):
    """
    删除知识文档
    """
    try:
        # 检查文档是否存在
        document_check = execute_query(
            """SELECT * FROM knowledge_documents WHERE document_id = %s""", 
            (document_id,)
        )
        
        if not document_check:
            return {
                "code": 404,
                "message": "文档不存在"
            }
        
        document_title = document_check[0]['title']
        file_path = document_check[0]['file_path']
        
        # 删除文档记录
        execute_update(
            """DELETE FROM knowledge_documents WHERE document_id = %s""", 
            (document_id,)
        )
        
        # 尝试删除物理文件
        if file_path:
            try:
                physical_path = DOCUMENT_UPLOAD_DIR / file_path
                if os.path.exists(physical_path):
                    os.remove(physical_path)
            except Exception as file_error:
                print(f"删除文件失败: {str(file_error)}")
        
        # 记录操作日志
        if admin_id:
            try:
                execute_update(
                    """
                    INSERT INTO operation_logs 
                        (admin_id, operation_type, operation_desc, created_at) 
                    VALUES 
                        (%s, %s, %s, NOW())
                    """, 
                    (admin_id, "删除文档", f"管理员删除了文档'{document_title}'(ID:{document_id})")
                )
            except Exception as log_error:
                print(f"记录操作日志失败: {str(log_error)}")
        
        return {
            "code": 200,
            "message": "删除文档成功"
        }
    except Exception as e:
        # 记录错误日志
        print(f"删除文档失败: {str(e)}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"删除文档失败: {str(e)}")

# 下载文档接口
@router.get("/admin/content/documents/{document_id}/download", tags=["内容管理"])
async def download_document(document_id: int):
    """
    下载文档文件
    """
    try:
        # 查询文档信息
        document = execute_query(
            """
            SELECT 
                document_id, title, file_path, file_type 
            FROM 
                knowledge_documents 
            WHERE 
                document_id = %s
            """, 
            (document_id,)
        )
        
        if not document:
            raise HTTPException(status_code=404, detail="文档不存在")
        
        file_path = document[0]['file_path']
        title = document[0]['title']
        
        if not file_path:
            raise HTTPException(status_code=404, detail="文档没有关联文件")
        
        physical_path = DOCUMENT_UPLOAD_DIR / file_path
        
        if not os.path.exists(physical_path):
            raise HTTPException(status_code=404, detail="文件不存在")
        
        # 更新下载计数
        execute_update(
            """
            UPDATE knowledge_documents 
            SET 
                download_count = download_count + 1 
            WHERE 
                document_id = %s
            """, 
            (document_id,)
        )
        
        # 返回文件作为响应
        return FileResponse(
            path=physical_path,
            filename=f"{title}.{document[0]['file_type'].lower()}",
            media_type="application/octet-stream"
        )
    except HTTPException:
        raise
    except Exception as e:
        # 记录错误日志
        print(f"下载文档失败: {str(e)}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"下载文档失败: {str(e)}")
