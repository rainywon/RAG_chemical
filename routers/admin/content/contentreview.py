# 引入 FastAPI 中的 APIRouter 和 HTTPException 模块，用于创建路由和处理异常
from fastapi import APIRouter, HTTPException, Depends, Query
# 引入 Pydantic 中的 BaseModel 类，用于定义请求体的数据结构和验证
from pydantic import BaseModel
# 从数据库模块导入 execute_query 和 execute_update 函数，用于执行查询和更新操作
from database import execute_query, execute_update
# 引入时间模块处理日期
from datetime import datetime, timedelta
# 引入 typing 模块中的 Optional 和 List 类型
from typing import Optional, List
# 引入管理员认证依赖函数
from routers.login import get_current_admin

# 初始化 APIRouter 实例，用于定义路由
router = APIRouter()

# 获取内容反馈列表接口
@router.get("/admin/content/feedback", tags=["内容管理"])
async def get_content_feedback(
    page: int = Query(1, description="页码"),
    page_size: int = Query(10, description="每页数量"),
    rating: Optional[int] = Query(None, description="评分筛选"),
    feedback_option: Optional[str] = Query(None, description="反馈类型筛选"),
    start_date: Optional[str] = Query(None, description="开始日期"),
    end_date: Optional[str] = Query(None, description="结束日期")
):
    """
    获取内容反馈列表，支持分页和筛选
    """
    try:
        # 计算偏移量
        offset = (page - 1) * page_size
        
        # 构建基础查询SQL
        query = """
            SELECT 
                id, rating, feedback, feedback_option, message, question, created_at
            FROM 
                content_feedbacks
            WHERE 1=1
        """
        params = []
        
        # 添加筛选条件
        if rating is not None:
            query += " AND rating = %s"
            params.append(rating)
        
        if feedback_option:
            query += " AND feedback_option = %s"
            params.append(feedback_option)
        
        if start_date:
            query += " AND DATE(created_at) >= %s"
            params.append(start_date)
        
        if end_date:
            query += " AND DATE(created_at) <= %s"
            params.append(end_date)
        
        # 查询符合条件的总记录数
        count_query = f"SELECT COUNT(*) as count FROM ({query}) as filtered_feedback"
        count_result = execute_query(count_query, tuple(params))
        total_count = count_result[0]['count'] if count_result else 0
        
        # 添加排序和分页
        query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
        params.append(page_size)
        params.append(offset)
        
        # 查询反馈列表
        feedback_list = execute_query(query, tuple(params))
        
        # 处理日期时间格式和缩短内容
        for feedback in feedback_list:
            feedback['created_at'] = feedback['created_at'].strftime("%Y-%m-%d %H:%M:%S") if feedback['created_at'] else None
            
            # 截断长消息和问题以避免过大的响应体
            if feedback['message'] and len(feedback['message']) > 300:
                feedback['message'] = feedback['message'][:300] + "..."
            if feedback['question'] and len(feedback['question']) > 300:
                feedback['question'] = feedback['question'][:300] + "..."
        
        return {
            "code": 200,
            "message": "获取反馈列表成功",
            "data": {
                "feedback": feedback_list,
                "total": total_count
            }
        }
    except Exception as e:
        # 记录错误日志
        print(f"获取反馈列表失败: {str(e)}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"获取反馈列表失败: {str(e)}")

# 获取反馈详情接口
@router.get("/admin/content/feedback/{feedback_id}", tags=["内容管理"])
async def get_feedback_detail(feedback_id: int):
    """
    获取反馈详细信息
    """
    try:
        # 查询反馈详情
        feedback_detail = execute_query(
            """
            SELECT 
                id, rating, feedback, feedback_option, message, question, created_at
            FROM 
                content_feedbacks
            WHERE 
                id = %s
            """, 
            (feedback_id,)
        )
        
        if not feedback_detail:
            return {
                "code": 404,
                "message": "反馈不存在"
            }
        
        # 处理日期时间格式
        feedback = feedback_detail[0]
        feedback['created_at'] = feedback['created_at'].strftime("%Y-%m-%d %H:%M:%S") if feedback['created_at'] else None
        
        return {
            "code": 200,
            "message": "获取反馈详情成功",
            "data": feedback
        }
    except Exception as e:
        # 记录错误日志
        print(f"获取反馈详情失败: {str(e)}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"获取反馈详情失败: {str(e)}")

# 删除反馈接口
@router.delete("/admin/content/feedback/{feedback_id}", tags=["内容管理"])
async def delete_feedback(feedback_id: int, admin_id: Optional[int] = None):
    """
    删除内容反馈
    """
    try:
        # 检查反馈是否存在
        feedback_check = execute_query(
            """SELECT * FROM content_feedbacks WHERE id = %s""", 
            (feedback_id,)
        )
        
        if not feedback_check:
            return {
                "code": 404,
                "message": "反馈不存在"
            }
        
        # 删除反馈记录
        execute_update(
            """DELETE FROM content_feedbacks WHERE id = %s""", 
            (feedback_id,)
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
                    (admin_id, "删除反馈", f"管理员删除了反馈(ID:{feedback_id})")
                )
            except Exception as log_error:
                print(f"记录操作日志失败: {str(log_error)}")
        
        return {
            "code": 200,
            "message": "删除反馈成功"
        }
    except Exception as e:
        # 记录错误日志
        print(f"删除反馈失败: {str(e)}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"删除反馈失败: {str(e)}")

# 获取反馈统计接口
@router.get("/admin/content/feedback/statistics", tags=["内容管理"])
async def get_feedback_statistics():
    """
    获取反馈统计信息
    """
    try:
        # 查询总反馈数
        total_count = execute_query("SELECT COUNT(*) as count FROM content_feedbacks")
        total = total_count[0]['count'] if total_count else 0
        
        # 查询各评分的反馈数
        rating_stats = execute_query(
            """
            SELECT 
                rating, COUNT(*) as count 
            FROM 
                content_feedbacks 
            GROUP BY 
                rating
            """
        )
        
        # 查询各反馈类型的反馈数
        option_stats = execute_query(
            """
            SELECT 
                feedback_option, COUNT(*) as count 
            FROM 
                content_feedbacks 
            GROUP BY 
                feedback_option
            """
        )
        
        # 查询最近7天每天的反馈数
        daily_stats = execute_query(
            """
            SELECT 
                DATE(created_at) as date, COUNT(*) as count 
            FROM 
                content_feedbacks 
            WHERE 
                created_at >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
            GROUP BY 
                DATE(created_at)
            ORDER BY 
                date
            """
        )
        
        # 处理日期格式
        for stat in daily_stats:
            if 'date' in stat and stat['date']:
                stat['date'] = stat['date'].strftime("%Y-%m-%d")
        
        return {
            "code": 200,
            "message": "获取反馈统计成功",
            "data": {
                "total": total,
                "rating_stats": rating_stats,
                "option_stats": option_stats,
                "daily_stats": daily_stats
            }
        }
    except Exception as e:
        # 记录错误日志
        print(f"获取反馈统计失败: {str(e)}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"获取反馈统计失败: {str(e)}")

# 添加示例反馈（用于测试）
@router.post("/admin/content/feedback/sample", tags=["内容管理"])
async def add_sample_feedback():
    """
    添加示例反馈数据（仅用于测试）
    """
    try:
        # 创建示例反馈
        sample_data = [
            {
                "rating": 3,
                "feedback": "回答内容不够详细，缺少具体的操作步骤",
                "feedback_option": "incomplete",
                "message": "化学品泄漏后，首先应该进行隔离，然后根据化学品的类型采取相应的处理措施。",
                "question": "如何处理化学品泄漏？"
            },
            {
                "rating": 2,
                "feedback": "回答内容与我的问题无关",
                "feedback_option": "irrelevant",
                "message": "工作场所通风是保障安全的重要措施，应该定期检查通风设备的运行状态。",
                "question": "如何处理氢氟酸烧伤？"
            },
            {
                "rating": 5,
                "feedback": "回答非常详细实用，符合标准操作流程",
                "feedback_option": "other",
                "message": "处理氢氟酸烧伤的步骤：1. 立即用大量清水冲洗至少15分钟；2. 涂抹2.5%的葡萄糖酸钙凝胶；3. 迅速就医，告知医生是氢氟酸烧伤。",
                "question": "如何处理氢氟酸烧伤？"
            }
        ]
        
        for data in sample_data:
            execute_update(
                """
                INSERT INTO content_feedbacks 
                    (rating, feedback, feedback_option, message, question, created_at) 
                VALUES 
                    (%s, %s, %s, %s, %s, NOW())
                """, 
                (
                    data["rating"], 
                    data["feedback"], 
                    data["feedback_option"], 
                    data["message"], 
                    data["question"]
                )
            )
        
        return {
            "code": 200,
            "message": "添加示例反馈成功",
            "data": {
                "count": len(sample_data)
            }
        }
    except Exception as e:
        # 记录错误日志
        print(f"添加示例反馈失败: {str(e)}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"添加示例反馈失败: {str(e)}")
