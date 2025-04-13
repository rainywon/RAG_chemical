# 引入 FastAPI 中的 APIRouter 和 HTTPException 模块，用于创建路由和处理异常
from fastapi import APIRouter, HTTPException, Depends, Query
# 引入 Pydantic 中的 BaseModel 类，用于定义请求体的数据结构和验证
from pydantic import BaseModel
# 从数据库模块导入 execute_query 和 execute_update 函数，用于执行查询和更新操作
from database import execute_query, execute_update
# 引入时间模块处理日期
from datetime import datetime
# 引入 typing 模块中的 Optional 和 List 类型
from typing import Optional, List, Literal
# 引入管理员认证依赖函数
from routers.login import get_current_admin

# 初始化 APIRouter 实例，用于定义路由
router = APIRouter()

# 定义应急方案类型
PlanType = Literal["fire", "leak", "poison", "other"]

# 定义应急方案请求体模型
class EmergencyPlanRequest(BaseModel):
    title: str
    plan_type: PlanType
    content: str
    is_published: Optional[int] = 1
    admin_id: Optional[int] = None

# 定义应急方案状态请求体模型
class PlanStatusRequest(BaseModel):
    is_published: int
    admin_id: Optional[int] = None

# 获取应急方案列表接口
@router.get("/admin/content/emergency-plans", tags=["内容管理"])
async def get_emergency_plans(
    page: int = Query(1, description="页码"),
    page_size: int = Query(10, description="每页数量"),
    title: Optional[str] = Query(None, description="标题筛选"),
    plan_type: Optional[str] = Query(None, description="类型筛选"),
    is_published: Optional[int] = Query(None, description="发布状态筛选")
):
    """
    获取应急处理方案列表，支持分页和筛选
    """
    try:
        # 计算偏移量
        offset = (page - 1) * page_size
        
        # 构建基础查询SQL
        query = """
            SELECT 
                plan_id, plan_type, title, content, is_published, 
                created_at, updated_at
            FROM 
                emergency_plans
            WHERE 1=1
        """
        params = []
        
        # 添加筛选条件
        if title:
            query += " AND title LIKE %s"
            params.append(f"%{title}%")
        
        if plan_type:
            query += " AND plan_type = %s"
            params.append(plan_type)
        
        if is_published is not None:
            query += " AND is_published = %s"
            params.append(is_published)
        
        # 查询符合条件的总记录数
        count_query = f"SELECT COUNT(*) as count FROM ({query}) as filtered_plans"
        count_result = execute_query(count_query, tuple(params))
        total_count = count_result[0]['count'] if count_result else 0
        
        # 添加排序和分页
        query += " ORDER BY updated_at DESC LIMIT %s OFFSET %s"
        params.append(page_size)
        params.append(offset)
        
        # 查询方案列表
        plan_list = execute_query(query, tuple(params))
        
        # 处理日期时间格式
        for plan in plan_list:
            plan['created_at'] = plan['created_at'].strftime("%Y-%m-%d %H:%M:%S") if plan['created_at'] else None
            plan['updated_at'] = plan['updated_at'].strftime("%Y-%m-%d %H:%M:%S") if plan['updated_at'] else None
        
        return {
            "code": 200,
            "message": "获取应急方案列表成功",
            "data": {
                "plans": plan_list,
                "total": total_count
            }
        }
    except Exception as e:
        # 记录错误日志
        print(f"获取应急方案列表失败: {str(e)}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"获取应急方案列表失败: {str(e)}")

# 创建应急方案接口
@router.post("/admin/content/emergency-plans", tags=["内容管理"])
async def create_emergency_plan(request: EmergencyPlanRequest):
    """
    创建新的应急处理方案
    """
    try:
        # 验证方案类型
        valid_types = ["fire", "leak", "poison", "other"]
        if request.plan_type not in valid_types:
            return {
                "code": 400,
                "message": "无效的方案类型"
            }
        
        # 插入方案记录
        insert_result = execute_update(
            """
            INSERT INTO emergency_plans 
                (plan_type, title, content, is_published) 
            VALUES 
                (%s, %s, %s, %s)
            """, 
            (
                request.plan_type, request.title, request.content, 
                request.is_published
            )
        )
        
        if not insert_result or 'lastrowid' not in insert_result:
            return {
                "code": 500,
                "message": "创建应急方案失败"
            }
        
        new_plan_id = insert_result['lastrowid']
        
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
                    (request.admin_id, "创建应急方案", f"管理员创建了新应急方案'{request.title}'")
                )
            except Exception as log_error:
                print(f"记录操作日志失败: {str(log_error)}")
        
        return {
            "code": 200,
            "message": "创建应急方案成功",
            "data": {
                "plan_id": new_plan_id
            }
        }
    except Exception as e:
        # 记录错误日志
        print(f"创建应急方案失败: {str(e)}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"创建应急方案失败: {str(e)}")

# 更新应急方案接口
@router.put("/admin/content/emergency-plans/{plan_id}", tags=["内容管理"])
async def update_emergency_plan(plan_id: int, request: EmergencyPlanRequest):
    """
    更新现有应急处理方案
    """
    try:
        # 检查方案是否存在
        plan_check = execute_query(
            """SELECT * FROM emergency_plans WHERE plan_id = %s""", 
            (plan_id,)
        )
        
        if not plan_check:
            return {
                "code": 404,
                "message": "应急方案不存在"
            }
        
        # 验证方案类型
        valid_types = ["fire", "leak", "poison", "other"]
        if request.plan_type not in valid_types:
            return {
                "code": 400,
                "message": "无效的方案类型"
            }
        
        # 更新方案记录
        execute_update(
            """
            UPDATE emergency_plans 
            SET 
                plan_type = %s, 
                title = %s, 
                content = %s, 
                is_published = %s,
                updated_at = NOW()
            WHERE 
                plan_id = %s
            """, 
            (
                request.plan_type, request.title, request.content, 
                request.is_published, plan_id
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
                    (request.admin_id, "更新应急方案", f"管理员更新了应急方案'{request.title}'(ID:{plan_id})")
                )
            except Exception as log_error:
                print(f"记录操作日志失败: {str(log_error)}")
        
        return {
            "code": 200,
            "message": "更新应急方案成功"
        }
    except Exception as e:
        # 记录错误日志
        print(f"更新应急方案失败: {str(e)}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"更新应急方案失败: {str(e)}")

# 更新应急方案发布状态接口
@router.put("/admin/content/emergency-plans/{plan_id}/status", tags=["内容管理"])
async def update_plan_status(plan_id: int, request: PlanStatusRequest):
    """
    更新应急方案的发布状态
    """
    try:
        # 检查方案是否存在
        plan_check = execute_query(
            """SELECT * FROM emergency_plans WHERE plan_id = %s""", 
            (plan_id,)
        )
        
        if not plan_check:
            return {
                "code": 404,
                "message": "应急方案不存在"
            }
        
        plan_title = plan_check[0]['title']
        old_status = plan_check[0]['is_published']
        new_status = request.is_published
        
        # 如果状态没有变化，直接返回成功
        if old_status == new_status:
            return {
                "code": 200,
                "message": "应急方案状态未变化"
            }
        
        # 更新方案状态
        execute_update(
            """
            UPDATE emergency_plans 
            SET 
                is_published = %s,
                updated_at = NOW()
            WHERE 
                plan_id = %s
            """, 
            (new_status, plan_id)
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
                    (request.admin_id, f"{action}应急方案", f"管理员{action}了应急方案'{plan_title}'(ID:{plan_id})")
                )
            except Exception as log_error:
                print(f"记录操作日志失败: {str(log_error)}")
        
        return {
            "code": 200,
            "message": "更新应急方案状态成功"
        }
    except Exception as e:
        # 记录错误日志
        print(f"更新应急方案状态失败: {str(e)}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"更新应急方案状态失败: {str(e)}")

# 删除应急方案接口
@router.delete("/admin/content/emergency-plans/{plan_id}", tags=["内容管理"])
async def delete_emergency_plan(plan_id: int, admin_id: Optional[int] = None):
    """
    删除应急处理方案
    """
    try:
        # 检查方案是否存在
        plan_check = execute_query(
            """SELECT * FROM emergency_plans WHERE plan_id = %s""", 
            (plan_id,)
        )
        
        if not plan_check:
            return {
                "code": 404,
                "message": "应急方案不存在"
            }
        
        plan_title = plan_check[0]['title']
        
        # 删除方案记录
        execute_update(
            """DELETE FROM emergency_plans WHERE plan_id = %s""", 
            (plan_id,)
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
                    (admin_id, "删除应急方案", f"管理员删除了应急方案'{plan_title}'(ID:{plan_id})")
                )
            except Exception as log_error:
                print(f"记录操作日志失败: {str(log_error)}")
        
        return {
            "code": 200,
            "message": "删除应急方案成功"
        }
    except Exception as e:
        # 记录错误日志
        print(f"删除应急方案失败: {str(e)}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"删除应急方案失败: {str(e)}")

# 获取应急方案详情接口
@router.get("/admin/content/emergency-plans/{plan_id}", tags=["内容管理"])
async def get_emergency_plan_detail(plan_id: int):
    """
    获取应急方案详细信息
    """
    try:
        # 查询方案详情
        plan_detail = execute_query(
            """
            SELECT 
                plan_id, plan_type, title, content, is_published, 
                created_at, updated_at
            FROM 
                emergency_plans
            WHERE 
                plan_id = %s
            """, 
            (plan_id,)
        )
        
        if not plan_detail:
            return {
                "code": 404,
                "message": "应急方案不存在"
            }
        
        # 处理日期时间格式
        plan = plan_detail[0]
        plan['created_at'] = plan['created_at'].strftime("%Y-%m-%d %H:%M:%S") if plan['created_at'] else None
        plan['updated_at'] = plan['updated_at'].strftime("%Y-%m-%d %H:%M:%S") if plan['updated_at'] else None
        
        return {
            "code": 200,
            "message": "获取应急方案详情成功",
            "data": plan
        }
    except Exception as e:
        # 记录错误日志
        print(f"获取应急方案详情失败: {str(e)}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"获取应急方案详情失败: {str(e)}")
