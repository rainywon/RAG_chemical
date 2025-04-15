# 引入 FastAPI 中的 APIRouter 和 HTTPException 模块，用于创建路由和处理异常
from fastapi import APIRouter, HTTPException, Depends
# 引入 Pydantic 中的 BaseModel 类，用于定义请求体的数据结构和验证
from pydantic import BaseModel
# 从数据库模块导入 execute_query 和 execute_update 函数，用于执行查询和更新操作
from database import execute_query
# 引入时间模块处理日期范围
from datetime import datetime, timedelta
# 引入JSON模块处理数据
import json
# 引入类型提示
from typing import List, Dict, Any, Optional
# 引入管理员认证依赖函数
from routers.user.login import get_current_admin

# 初始化 APIRouter 实例，用于定义路由
router = APIRouter()

# 定义日期范围请求模型
class ActivityTrendRequest(BaseModel):
    start_date: Optional[str] = None  # 开始日期，格式：YYYY-MM-DD
    end_date: Optional[str] = None  # 结束日期，格式：YYYY-MM-DD
    time_unit: str = "day"  # 时间单位，可选值：day, week, month

# 获取用户活跃度统计数据
@router.get("/admin/user/activity/stats", tags=["用户活跃度"])
async def get_activity_stats():
    """
    获取用户活跃度统计数据，包括活跃用户数和当前在线用户数
    """
    try:
        # 1. 活跃用户数（过去30天内有登录记录或有发送消息的用户）
        active_users_query = """
            SELECT COUNT(DISTINCT user_id) as count 
            FROM (
                SELECT DISTINCT user_id FROM user_tokens 
                WHERE created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
                UNION
                SELECT DISTINCT u.user_id 
                FROM chat_sessions cs
                JOIN users u ON cs.token = u.mobile
                JOIN chat_messages cm ON cs.id = cm.session_id
                WHERE cm.message_type = 'user' 
                AND cm.created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
            ) as active_users
        """
        active_users_result = execute_query(active_users_query)
        active_users = active_users_result[0]['count'] if active_users_result else 0
        
        # 2. 当前在线用户数（有效的且未过期的token数量）
        online_users_result = execute_query(
            """SELECT COUNT(DISTINCT user_id) as count FROM user_tokens 
               WHERE is_valid = 1 AND expire_at > NOW()"""
        )
        online_users = online_users_result[0]['count'] if online_users_result else 0
        
        # 3. 新增用户数（过去7天注册的用户）
        new_users_result = execute_query(
            """SELECT COUNT(*) as count FROM users 
               WHERE register_time >= DATE_SUB(NOW(), INTERVAL 7 DAY)"""
        )
        new_users = new_users_result[0]['count'] if new_users_result else 0
        
        return {
            "code": 200,
            "message": "获取用户活跃度统计数据成功",
            "data": {
                "active_users": active_users,
                "online_users": online_users,
                "new_users": new_users
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取用户活跃度统计数据失败: {str(e)}")

# 获取用户活跃趋势数据
@router.post("/admin/user/activity/trend", tags=["用户活跃度"])
async def get_activity_trend(request: ActivityTrendRequest):
    """
    获取指定日期范围内的用户活跃趋势数据
    """
    try:
        # 处理日期范围
        end_date = datetime.now()
        if request.end_date:
            end_date = datetime.strptime(request.end_date, "%Y-%m-%d")
        
        # 确定开始日期
        start_date = None
        if request.start_date:
            start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        else:
            # 根据时间单位设置默认日期范围
            if request.time_unit == "day":
                start_date = end_date - timedelta(days=14)  # 默认显示两周
            elif request.time_unit == "week":
                start_date = end_date - timedelta(weeks=12)  # 默认显示12周
            elif request.time_unit == "month":
                start_date = end_date - timedelta(days=365)  # 默认显示近一年（12个月）
            else:
                start_date = end_date - timedelta(days=14)  # 默认两周
        
        # 生成日期列表和数据
        date_list = []
        active_user_data = []
        new_user_data = []
        
        # 根据时间单位生成不同的SQL和日期列表
        if request.time_unit == "day":
            # 日粒度查询
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime("%Y-%m-%d")
                date_list.append(current_date.strftime("%m/%d"))  # 格式化为月/日
                
                # 查询当天的活跃用户数（有登录记录或发送消息的用户）
                active_users_query = """
                    SELECT COUNT(DISTINCT user_id) as count 
                    FROM (
                        SELECT DISTINCT user_id FROM user_tokens 
                        WHERE DATE(created_at) = %s
                        UNION
                        SELECT DISTINCT u.user_id 
                        FROM chat_sessions cs
                        JOIN users u ON cs.token = u.mobile
                        JOIN chat_messages cm ON cs.id = cm.session_id
                        WHERE cm.message_type = 'user' 
                        AND DATE(cm.created_at) = %s
                    ) as active_users
                """
                active_users_result = execute_query(active_users_query, (date_str, date_str))
                active_users = active_users_result[0]['count'] if active_users_result else 0
                active_user_data.append(active_users)
                
                # 查询当天的新增用户数
                new_users_result = execute_query(
                    """SELECT COUNT(*) as count FROM users 
                       WHERE DATE(register_time) = %s""",
                    (date_str,)
                )
                new_users = new_users_result[0]['count'] if new_users_result else 0
                new_user_data.append(new_users)
                
                current_date += timedelta(days=1)
                
        elif request.time_unit == "week":
            # 周粒度查询
            current_date = start_date
            week_count = 1
            while current_date <= end_date:
                week_end = current_date + timedelta(days=6)
                date_list.append(f"第{week_count}周")
                
                # 查询本周的活跃用户数（有登录记录或发送消息的用户）
                active_users_query = """
                    SELECT COUNT(DISTINCT user_id) as count 
                    FROM (
                        SELECT DISTINCT user_id FROM user_tokens 
                        WHERE created_at >= %s AND created_at <= %s
                        UNION
                        SELECT DISTINCT u.user_id 
                        FROM chat_sessions cs
                        JOIN users u ON cs.token = u.mobile
                        JOIN chat_messages cm ON cs.id = cm.session_id
                        WHERE cm.message_type = 'user' 
                        AND cm.created_at >= %s AND cm.created_at <= %s
                    ) as active_users
                """
                active_users_result = execute_query(active_users_query, (current_date, week_end, current_date, week_end))
                active_users = active_users_result[0]['count'] if active_users_result else 0
                active_user_data.append(active_users)
                
                # 查询本周的新增用户数
                new_users_result = execute_query(
                    """SELECT COUNT(*) as count FROM users 
                       WHERE register_time >= %s AND register_time <= %s""",
                    (current_date, week_end)
                )
                new_users = new_users_result[0]['count'] if new_users_result else 0
                new_user_data.append(new_users)
                
                current_date += timedelta(days=7)
                week_count += 1
                
        elif request.time_unit == "month":
            # 月粒度查询
            current_date = start_date.replace(day=1)  # 从月初开始
            while current_date <= end_date:
                # 获取下个月的第一天
                if current_date.month == 12:
                    next_month = current_date.replace(year=current_date.year+1, month=1)
                else:
                    next_month = current_date.replace(month=current_date.month+1)
                
                # 本月最后一天
                month_end = next_month - timedelta(days=1)
                
                date_list.append(f"{current_date.month}月")
                
                # 查询本月的活跃用户数（有登录记录或发送消息的用户）
                active_users_query = """
                    SELECT COUNT(DISTINCT user_id) as count 
                    FROM (
                        SELECT DISTINCT user_id FROM user_tokens 
                        WHERE created_at >= %s AND created_at <= %s
                        UNION
                        SELECT DISTINCT u.user_id 
                        FROM chat_sessions cs
                        JOIN users u ON cs.token = u.mobile
                        JOIN chat_messages cm ON cs.id = cm.session_id
                        WHERE cm.message_type = 'user' 
                        AND cm.created_at >= %s AND cm.created_at <= %s
                    ) as active_users
                """
                active_users_result = execute_query(active_users_query, (current_date, month_end, current_date, month_end))
                active_users = active_users_result[0]['count'] if active_users_result else 0
                active_user_data.append(active_users)
                
                # 查询本月的新增用户数
                new_users_result = execute_query(
                    """SELECT COUNT(*) as count FROM users 
                       WHERE register_time >= %s AND register_time <= %s""",
                    (current_date, month_end)
                )
                new_users = new_users_result[0]['count'] if new_users_result else 0
                new_user_data.append(new_users)
                
                current_date = next_month
        
        return {
            "code": 200,
            "message": "获取用户活跃趋势数据成功",
            "data": {
                "dates": date_list,
                "active_user_data": active_user_data,
                "new_user_data": new_user_data
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取用户活跃趋势数据失败: {str(e)}")

# 获取最近登录记录
@router.get("/admin/user/logins", tags=["用户活跃度"])
async def get_recent_logins(
    page: int = 1, 
    page_size: int = 10,
):
    """
    获取最近的用户登录记录，分页展示
    """
    try:
        # 计算偏移量
        offset = (page - 1) * page_size
        
        # 查询最近登录记录
        recent_logins_query = """
            SELECT 
                ut.user_id,
                u.mobile,
                ut.created_at as login_time
            FROM 
                user_tokens ut
            JOIN 
                users u ON ut.user_id = u.user_id
            ORDER BY 
                ut.created_at DESC
            LIMIT %s OFFSET %s
        """
        
        recent_logins_result = execute_query(recent_logins_query, (page_size, offset))
        
        # 获取总登录记录数
        total_count_result = execute_query(
            """SELECT COUNT(*) as count FROM user_tokens"""
        )
        total_count = total_count_result[0]['count'] if total_count_result else 0
        
        # 处理日期时间格式
        logins = []
        for login in recent_logins_result:
            logins.append({
                "user_id": login['user_id'],
                "mobile": login['mobile'],
                "login_time": login['login_time'].strftime("%Y-%m-%d %H:%M:%S") if login['login_time'] else ""
            })
        
        return {
            "code": 200,
            "message": "获取最近登录记录成功",
            "data": {
                "logins": logins,
                "total": total_count
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取最近登录记录失败: {str(e)}")
