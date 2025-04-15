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
class DateRangeRequest(BaseModel):
    start_date: Optional[str] = None  # 开始日期，格式：YYYY-MM-DD
    end_date: Optional[str] = None  # 结束日期，格式：YYYY-MM-DD
    time_unit: str = "day"  # 时间单位，可选值：day, week, month

# 获取统计概览数据
@router.get("/admin/conversation/stats", tags=["会话统计"])
async def get_conversation_stats():
    """
    获取对话统计概览数据，包括总对话数、总消息数、活跃用户数等
    """
    try:
        # 获取统计概览数据
        # 1. 总对话数
        total_sessions_result = execute_query(
            """SELECT COUNT(*) as count FROM chat_sessions"""
        )
        total_sessions = total_sessions_result[0]['count'] if total_sessions_result else 0
        
        # 2. 总消息数
        total_messages_result = execute_query(
            """SELECT COUNT(*) as count FROM chat_messages"""
        )
        total_messages = total_messages_result[0]['count'] if total_messages_result else 0
        
        # 3. 活跃用户数（过去30天内有对话的用户）
        active_users_result = execute_query(
            """SELECT COUNT(DISTINCT token) as count FROM chat_sessions 
               WHERE updated_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)"""
        )
        active_users = active_users_result[0]['count'] if active_users_result else 0
        
        # 4. 平均每次对话的消息数
        avg_messages_per_session = 0
        if total_sessions > 0:
            avg_messages_per_session = total_messages / total_sessions
        
        return {
            "code": 200,
            "message": "获取对话统计数据成功",
            "data": {
                "total_sessions": total_sessions,
                "total_messages": total_messages,
                "active_users": active_users,
                "avg_messages_per_session": avg_messages_per_session
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取对话统计数据失败: {str(e)}")

# 获取对话趋势数据
@router.post("/admin/conversation/trend", tags=["会话统计"])
async def get_conversation_trend(request: DateRangeRequest):
    """
    获取指定日期范围内的对话趋势数据
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
        
        # 生成日期列表
        date_list = []
        session_data = []
        message_data = []
        
        # 根据时间单位生成不同的SQL和日期列表
        if request.time_unit == "day":
            # 日粒度查询
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime("%Y-%m-%d")
                date_list.append(current_date.strftime("%m/%d"))  # 格式化为月/日
                
                # 查询当天的会话数
                session_count_result = execute_query(
                    """SELECT COUNT(*) as count FROM chat_sessions 
                       WHERE DATE(created_at) = %s""",
                    (date_str,)
                )
                session_count = session_count_result[0]['count'] if session_count_result else 0
                session_data.append(session_count)
                
                # 查询当天的消息数
                message_count_result = execute_query(
                    """SELECT COUNT(*) as count FROM chat_messages 
                       WHERE DATE(created_at) = %s""",
                    (date_str,)
                )
                message_count = message_count_result[0]['count'] if message_count_result else 0
                message_data.append(message_count)
                
                current_date += timedelta(days=1)
                
        elif request.time_unit == "week":
            # 周粒度查询
            current_date = start_date
            week_count = 1
            while current_date <= end_date:
                week_end = current_date + timedelta(days=6)
                date_list.append(f"第{week_count}周")
                
                # 查询本周的会话数
                session_count_result = execute_query(
                    """SELECT COUNT(*) as count FROM chat_sessions 
                       WHERE created_at >= %s AND created_at <= %s""",
                    (current_date, week_end)
                )
                session_count = session_count_result[0]['count'] if session_count_result else 0
                session_data.append(session_count)
                
                # 查询本周的消息数
                message_count_result = execute_query(
                    """SELECT COUNT(*) as count FROM chat_messages 
                       WHERE created_at >= %s AND created_at <= %s""",
                    (current_date, week_end)
                )
                message_count = message_count_result[0]['count'] if message_count_result else 0
                message_data.append(message_count)
                
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
                
                # 查询本月的会话数
                session_count_result = execute_query(
                    """SELECT COUNT(*) as count FROM chat_sessions 
                       WHERE created_at >= %s AND created_at <= %s""",
                    (current_date, month_end)
                )
                session_count = session_count_result[0]['count'] if session_count_result else 0
                session_data.append(session_count)
                
                # 查询本月的消息数
                message_count_result = execute_query(
                    """SELECT COUNT(*) as count FROM chat_messages 
                       WHERE created_at >= %s AND created_at <= %s""",
                    (current_date, month_end)
                )
                message_count = message_count_result[0]['count'] if message_count_result else 0
                message_data.append(message_count)
                
                current_date = next_month
        
        return {
            "code": 200,
            "message": "获取对话趋势数据成功",
            "data": {
                "dates": date_list,
                "session_data": session_data,
                "message_data": message_data
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取对话趋势数据失败: {str(e)}")

# 获取最近对话列表
@router.get("/admin/conversation/recent", tags=["会话统计"])
async def get_recent_conversations(
    page: int = 1, 
    page_size: int = 10,
):
    """
    获取最近的对话列表，分页展示
    """
    try:
        # 计算偏移量
        offset = (page - 1) * page_size
        
        # 查询最近对话
        recent_conversations_result = execute_query(
            """SELECT 
                  cs.id, 
                  cs.title, 
                  cs.token, 
                  cs.created_at,
                  cs.updated_at,
                  COUNT(cm.id) as message_count
               FROM 
                  chat_sessions cs
               LEFT JOIN 
                  chat_messages cm ON cs.id = cm.session_id
               GROUP BY 
                  cs.id
               ORDER BY 
                  cs.updated_at DESC
               LIMIT %s OFFSET %s""",
            (page_size, offset)
        )
        
        # 获取总对话数
        total_count_result = execute_query(
            """SELECT COUNT(*) as count FROM chat_sessions"""
        )
        total_count = total_count_result[0]['count'] if total_count_result else 0
        
        # 处理日期时间格式
        conversations = []
        for conv in recent_conversations_result:
            conversations.append({
                "id": conv['id'],
                "title": conv['title'],
                "token": conv['token'],
                "message_count": conv['message_count'],
                "created_at": conv['created_at'].strftime("%Y-%m-%d %H:%M:%S"),
                "updated_at": conv['updated_at'].strftime("%Y-%m-%d %H:%M:%S") if conv['updated_at'] else None
            })
        
        return {
            "code": 200,
            "message": "获取最近对话列表成功",
            "data": {
                "conversations": conversations,
                "total": total_count,
                "page": page,
                "page_size": page_size
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取最近对话列表失败: {str(e)}") 