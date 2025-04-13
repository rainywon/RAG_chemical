# 引入 FastAPI 中的 APIRouter 和 HTTPException 模块，用于创建路由和处理异常
from fastapi import APIRouter, HTTPException, Depends, Query
# 从数据库模块导入 execute_query 和 execute_update 函数，用于执行查询和更新操作
from database import execute_query, execute_update
# 引入时间模块处理日期范围
from datetime import datetime, timedelta
# 引入 typing 模块中的 Optional 类型
from typing import Optional, Dict

# 初始化 APIRouter 实例，用于定义路由
router = APIRouter()

# 获取内容评价统计概览
@router.get("/admin/feedback/content-rating-summary", tags=["内容评价"])
async def get_content_rating_summary(
    rating: Optional[int] = Query(None, description="评分筛选"),
    feedback_option: Optional[str] = Query(None, description="反馈选项筛选"),
    start_date: Optional[str] = Query(None, description="开始日期"),
    end_date: Optional[str] = Query(None, description="结束日期"),
    admin_id: Optional[int] = Query(None, description="管理员ID")
):
    """
    获取内容评价统计概览，包括平均评分、评分分布和反馈选项分布
    """
    try:
        # 构建条件子句
        where_clause = "1=1"
        params = []
        
        if rating is not None:
            where_clause += " AND rating = %s"
            params.append(rating)
        
        if feedback_option:
            where_clause += " AND feedback_option = %s"
            params.append(feedback_option)
        
        if start_date:
            where_clause += " AND DATE(created_at) >= %s"
            params.append(start_date)
        
        if end_date:
            where_clause += " AND DATE(created_at) <= %s"
            params.append(end_date)
        
        # 查询平均评分
        avg_query = f"""
            SELECT 
                ROUND(AVG(rating), 2) as average_rating
            FROM 
                content_feedbacks
            WHERE 
                {where_clause}
        """
        
        avg_result = execute_query(avg_query, tuple(params))
        avg_rating = avg_result[0]['average_rating'] if avg_result and avg_result[0]['average_rating'] is not None else 0
        
        # 查询评分分布
        dist_query = f"""
            SELECT 
                rating,
                COUNT(*) as count
            FROM 
                content_feedbacks
            WHERE 
                {where_clause}
            GROUP BY 
                rating
            ORDER BY 
                rating DESC
        """
        
        dist_results = execute_query(dist_query, tuple(params))
        
        # 构建评分分布字典
        rating_distribution = {}
        for i in range(1, 6):
            rating_distribution[i] = 0
        
        for item in dist_results:
            rating_distribution[item['rating']] = item['count']
        
        # 查询反馈选项分布
        option_query = f"""
            SELECT 
                feedback_option,
                COUNT(*) as count
            FROM 
                content_feedbacks
            WHERE 
                {where_clause}
            GROUP BY 
                feedback_option
        """
        
        option_results = execute_query(option_query, tuple(params))
        
        # 构建反馈选项分布字典
        feedback_distribution = {}
        for item in option_results:
            if item['feedback_option']:
                feedback_distribution[item['feedback_option']] = item['count']
        
        # 记录操作日志 (如果提供了管理员ID)
        if admin_id:
            try:
                execute_update(
                    """INSERT INTO operation_logs (admin_id, operation_type, operation_desc, created_at) 
                       VALUES (%s, %s, %s, NOW())""", 
                    (admin_id, "查看内容评价统计", f"管理员{admin_id}查看内容评价统计数据")
                )
            except Exception as log_error:
                # 仅记录日志错误，不影响主流程
                print(f"记录操作日志失败: {str(log_error)}")
        
        return {
            "code": 200,
            "message": "获取内容评价统计成功",
            "data": {
                "average_rating": float(avg_rating),
                "rating_distribution": rating_distribution,
                "feedback_distribution": feedback_distribution
            }
        }
    except Exception as e:
        # 记录错误日志
        print(f"获取内容评价统计失败: {str(e)}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"获取内容评价统计失败: {str(e)}")

# 获取内容评价列表
@router.get("/admin/feedback/content-rating-list", tags=["内容评价"])
async def get_content_rating_list(
    page: int = Query(1, description="页码"),
    page_size: int = Query(10, description="每页数量"),
    rating: Optional[int] = Query(None, description="评分筛选"),
    feedback_option: Optional[str] = Query(None, description="反馈选项筛选"),
    start_date: Optional[str] = Query(None, description="开始日期"),
    end_date: Optional[str] = Query(None, description="结束日期"),
    admin_id: Optional[int] = Query(None, description="管理员ID")
):
    """
    获取内容评价列表，支持分页和筛选
    """
    try:
        # 计算偏移量
        offset = (page - 1) * page_size
        
        # 构建条件子句
        where_clause = "1=1"
        params = []
        
        if rating is not None:
            where_clause += " AND rating = %s"
            params.append(rating)
        
        if feedback_option:
            where_clause += " AND feedback_option = %s"
            params.append(feedback_option)
        
        if start_date:
            where_clause += " AND DATE(created_at) >= %s"
            params.append(start_date)
        
        if end_date:
            where_clause += " AND DATE(created_at) <= %s"
            params.append(end_date)
        
        # 查询总数
        count_query = f"""
            SELECT 
                COUNT(*) as count
            FROM 
                content_feedbacks
            WHERE 
                {where_clause}
        """
        
        count_result = execute_query(count_query, tuple(params))
        total_count = count_result[0]['count'] if count_result else 0
        
        # 查询列表数据
        list_query = f"""
            SELECT 
                id, rating, feedback, feedback_option, message, question, created_at
            FROM 
                content_feedbacks
            WHERE 
                {where_clause}
            ORDER BY 
                created_at DESC
            LIMIT %s OFFSET %s
        """
        
        list_params = list(params)
        list_params.append(page_size)
        list_params.append(offset)
        
        list_results = execute_query(list_query, tuple(list_params))
        
        # 处理日期时间格式
        for item in list_results:
            item['created_at'] = item['created_at'].strftime("%Y-%m-%d %H:%M:%S") if item['created_at'] else ""
            
            # 限制内容长度，前端展示用
            if item.get('message') and len(item['message']) > 300:
                item['message_preview'] = item['message'][:300] + "..."
            else:
                item['message_preview'] = item.get('message', '')
                
            if item.get('question') and len(item['question']) > 300:
                item['question_preview'] = item['question'][:300] + "..."
            else:
                item['question_preview'] = item.get('question', '')
        
        # 记录操作日志 (如果提供了管理员ID)
        if admin_id:
            try:
                execute_update(
                    """INSERT INTO operation_logs (admin_id, operation_type, operation_desc, created_at) 
                       VALUES (%s, %s, %s, NOW())""", 
                    (admin_id, "查看内容评价列表", f"管理员{admin_id}查看内容评价列表")
                )
            except Exception as log_error:
                # 仅记录日志错误，不影响主流程
                print(f"记录操作日志失败: {str(log_error)}")
        
        return {
            "code": 200,
            "message": "获取内容评价列表成功",
            "data": {
                "list": list_results,
                "total": total_count
            }
        }
    except Exception as e:
        # 记录错误日志
        print(f"获取内容评价列表失败: {str(e)}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"获取内容评价列表失败: {str(e)}")
