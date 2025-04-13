# 引入 FastAPI 中的 APIRouter 和 HTTPException 模块，用于创建路由和处理异常
from fastapi import APIRouter, HTTPException, Depends, Query
# 引入 Pydantic 中的 BaseModel 类，用于定义请求体的数据结构和验证
from pydantic import BaseModel
# 从数据库模块导入 execute_query 和 execute_update 函数，用于执行查询和更新操作
from database import execute_query, execute_update
# 引入时间模块处理日期范围
from datetime import datetime
# 引入 typing 模块中的 Optional 和 List 类型
from typing import Optional, List, Dict, Any
# 引入管理员认证依赖函数
from routers.login import get_current_admin

# 初始化 APIRouter 实例，用于定义路由
router = APIRouter()

# 定义请求体的模型，用于新增和更新分类
class CategoryRequest(BaseModel):
    category_name: str
    parent_id: Optional[int] = None
    icon: Optional[str] = None
    description: Optional[str] = None
    sort_order: Optional[int] = 0
    admin_id: Optional[int] = None

# 获取所有分类列表接口
@router.get("/admin/content/categories", tags=["内容管理"])
async def get_categories():
    """
    获取所有分类列表，包括其层级关系
    """
    try:
        # 查询所有分类
        query = """
            SELECT 
                category_id, category_name, parent_id, icon, description, sort_order
            FROM 
                knowledge_categories
            ORDER BY 
                parent_id IS NULL DESC, sort_order ASC, category_id ASC
        """
        
        categories = execute_query(query)
        
        # 构建分类树形结构
        root_categories = []
        category_map = {}
        
        # 先创建所有分类的映射
        for category in categories:
            category_id = category['category_id']
            category_map[category_id] = category
            category['children'] = []
        
        # 构建树形结构
        for category in categories:
            parent_id = category['parent_id']
            if parent_id is None:
                # 这是一个根分类
                root_categories.append(category)
            else:
                # 将此分类添加到其父分类的子分类列表中
                if parent_id in category_map:
                    category_map[parent_id]['children'].append(category)
        
        return {
            "code": 200,
            "message": "获取分类列表成功",
            "data": {
                "categories": categories if not root_categories else root_categories
            }
        }
    except Exception as e:
        # 记录错误日志
        print(f"获取分类列表失败: {str(e)}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"获取分类列表失败: {str(e)}")

# 创建新分类接口
@router.post("/admin/content/categories", tags=["内容管理"])
async def create_category(request: CategoryRequest):
    """
    创建新的知识分类
    """
    try:
        # 检查父分类是否存在
        if request.parent_id is not None:
            parent_check = execute_query(
                """SELECT * FROM knowledge_categories WHERE category_id = %s""", 
                (request.parent_id,)
            )
            
            if not parent_check:
                return {
                    "code": 404,
                    "message": "上级分类不存在"
                }
        
        # 插入新分类
        insert_result = execute_update(
            """
            INSERT INTO knowledge_categories 
                (category_name, parent_id, icon, description, sort_order) 
            VALUES 
                (%s, %s, %s, %s, %s)
            """, 
            (
                request.category_name, 
                request.parent_id, 
                request.icon, 
                request.description, 
                request.sort_order
            )
        )
        
        if not insert_result or 'lastrowid' not in insert_result:
            return {
                "code": 500,
                "message": "创建分类失败"
            }
        
        new_category_id = insert_result['lastrowid']
        
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
                    (request.admin_id, "创建分类", f"管理员创建了新分类'{request.category_name}'")
                )
            except Exception as log_error:
                print(f"记录操作日志失败: {str(log_error)}")
        
        return {
            "code": 200,
            "message": "创建分类成功",
            "data": {
                "category_id": new_category_id
            }
        }
    except Exception as e:
        # 记录错误日志
        print(f"创建分类失败: {str(e)}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"创建分类失败: {str(e)}")

# 更新分类接口
@router.put("/admin/content/categories/{category_id}", tags=["内容管理"])
async def update_category(category_id: int, request: CategoryRequest):
    """
    更新现有知识分类
    """
    try:
        # 检查分类是否存在
        category_check = execute_query(
            """SELECT * FROM knowledge_categories WHERE category_id = %s""", 
            (category_id,)
        )
        
        if not category_check:
            return {
                "code": 404,
                "message": "分类不存在"
            }
        
        # 检查父分类是否存在
        if request.parent_id is not None:
            parent_check = execute_query(
                """SELECT * FROM knowledge_categories WHERE category_id = %s""", 
                (request.parent_id,)
            )
            
            if not parent_check:
                return {
                    "code": 404,
                    "message": "上级分类不存在"
                }
            
            # 检查是否将分类设置为其自身的子分类（防止循环）
            if request.parent_id == category_id:
                return {
                    "code": 400,
                    "message": "不能将分类设置为自身的子分类"
                }
            
            # TODO: 更高级的检查，例如不能将分类设置为其子分类的子分类
        
        # 更新分类
        execute_update(
            """
            UPDATE knowledge_categories 
            SET 
                category_name = %s, 
                parent_id = %s, 
                icon = %s, 
                description = %s, 
                sort_order = %s
            WHERE 
                category_id = %s
            """, 
            (
                request.category_name, 
                request.parent_id, 
                request.icon, 
                request.description, 
                request.sort_order,
                category_id
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
                    (request.admin_id, "更新分类", f"管理员更新了分类'{request.category_name}'(ID:{category_id})")
                )
            except Exception as log_error:
                print(f"记录操作日志失败: {str(log_error)}")
        
        return {
            "code": 200,
            "message": "更新分类成功"
        }
    except Exception as e:
        # 记录错误日志
        print(f"更新分类失败: {str(e)}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"更新分类失败: {str(e)}")

# 删除分类接口
@router.delete("/admin/content/categories/{category_id}", tags=["内容管理"])
async def delete_category(category_id: int, admin_id: Optional[int] = None):
    """
    删除知识分类
    """
    try:
        # 检查分类是否存在
        category_check = execute_query(
            """SELECT * FROM knowledge_categories WHERE category_id = %s""", 
            (category_id,)
        )
        
        if not category_check:
            return {
                "code": 404,
                "message": "分类不存在"
            }
        
        category_name = category_check[0]['category_name']
        
        # 检查是否有子分类
        child_check = execute_query(
            """SELECT * FROM knowledge_categories WHERE parent_id = %s LIMIT 1""", 
            (category_id,)
        )
        
        if child_check:
            return {
                "code": 400,
                "message": "该分类下存在子分类，请先删除子分类"
            }
        
        # 检查分类下是否有文档
        doc_check = execute_query(
            """SELECT * FROM knowledge_documents WHERE category_id = %s LIMIT 1""", 
            (category_id,)
        )
        
        if doc_check:
            return {
                "code": 400,
                "message": "该分类下存在文档，无法删除"
            }
        
        # 删除分类
        execute_update(
            """DELETE FROM knowledge_categories WHERE category_id = %s""", 
            (category_id,)
        )
        
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
                    (admin_id, "删除分类", f"管理员删除了分类'{category_name}'(ID:{category_id})")
                )
            except Exception as log_error:
                print(f"记录操作日志失败: {str(log_error)}")
        
        return {
            "code": 200,
            "message": "删除分类成功"
        }
    except Exception as e:
        # 记录错误日志
        print(f"删除分类失败: {str(e)}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"删除分类失败: {str(e)}")
