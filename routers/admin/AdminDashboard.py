from fastapi import APIRouter, HTTPException, Depends, Query, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from database import execute_query, execute_update
from datetime import datetime, timedelta
from typing import Optional, List
from routers.user.login import get_current_admin
import os
from pathlib import Path

# 初始化安全认证
security = HTTPBearer()

# 从请求中提取管理员ID的辅助函数
async def get_admin_id_from_request(request: Request):
    """
    从请求中的Authorization头部获取管理员ID
    如果无法获取到有效的管理员ID，返回None
    """
    try:
        # 从Authorization头获取token
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return None
        
        token = auth_header.split(' ')[1]
        
        # 查询token对应的管理员ID
        token_result = execute_query(
            """SELECT admin_id 
               FROM admin_tokens 
               WHERE token = %s AND is_valid = 1 AND expire_at > NOW()""",
            (token,)
        )
        
        if not token_result or len(token_result) == 0:
            return None
        
        return token_result[0]['admin_id']
    except Exception as e:
        print(f"从请求中获取管理员ID失败: {str(e)}")
        return None

# 初始化 APIRouter 实例，用于定义路由
router = APIRouter()

@router.get("/admin/dashboard/stats", tags=["仪表盘"])
async def get_dashboard_stats(request: Request):
    """
    获取仪表盘统计数据
    """
    try:
        # 从token中获取管理员ID
        try:
            admin_id = await get_admin_id_from_request(request)
        except Exception as auth_error:
            print(f"获取管理员ID失败: {str(auth_error)}")
            admin_id = None
        
        # 用户统计
        try:
            user_stats = execute_query(
                """SELECT 
                    COUNT(*) as total_users,
                    SUM(CASE WHEN DATE(register_time) = CURDATE() THEN 1 ELSE 0 END) as new_users
                   FROM users""", 
                ()
            )
        except Exception as e:
            print(f"获取用户统计失败: {str(e)}")
            user_stats = []
        
        # 内容统计
        try:
            # 从文件系统获取内容统计数据，不再从数据库中查询
            # 设置相关文件路径，确保与content目录下的代码保持一致
            KNOWLEDGE_BASE_PATH = r"C:\Users\coins\Desktop\chunks"
            SAFETY_DOCUMENT_PATH = r"C:\Users\coins\Desktop\chemical\chemical_rag\data\标准性文件"
            EMERGENCY_PLAN_PATH = r"C:\Users\coins\Desktop\chemical\chemical_rag\data\标准性文件"
            
            # 确保路径存在
            os.makedirs(KNOWLEDGE_BASE_PATH, exist_ok=True)
            os.makedirs(SAFETY_DOCUMENT_PATH, exist_ok=True)
            os.makedirs(EMERGENCY_PLAN_PATH, exist_ok=True)
            
            # 统计各类文件数量
            import glob
            
            # 知识库文件数量 - 使用glob统计Excel文件数量
            knowledge_files = glob.glob(os.path.join(KNOWLEDGE_BASE_PATH, "*.xlsx")) + glob.glob(os.path.join(KNOWLEDGE_BASE_PATH, "*.xls"))
            knowledge_count = len(knowledge_files)
            
            # 安全资料库文件数量 - 仅统计PDF、DOC、DOCX文件
            safety_files = []
            all_safety_files = glob.glob(os.path.join(SAFETY_DOCUMENT_PATH, "*.*"))
            for file_path in all_safety_files:
                file_ext = Path(file_path).suffix.lower()
                if file_ext in ['.pdf', '.doc', '.docx']:
                    safety_files.append(file_path)
            safety_count = len(safety_files)
            
            # 应急预案文件数量 - 仅统计PDF、DOC、DOCX文件
            emergency_files = []
            all_emergency_files = glob.glob(os.path.join(EMERGENCY_PLAN_PATH, "*.*"))
            for file_path in all_emergency_files:
                file_ext = Path(file_path).suffix.lower()
                if file_ext in ['.pdf', '.doc', '.docx']:
                    emergency_files.append(file_path)
            emergency_count = len(emergency_files)
            
            # 构建统计结果
            content_stats = [{
                'knowledge_count': knowledge_count,
                'safety_count': safety_count,
                'emergency_count': emergency_count
            }]
        except Exception as e:
            print(f"获取内容统计失败: {str(e)}")
            content_stats = [{'knowledge_count': 0, 'safety_count': 0, 'emergency_count': 0}]
        
        # 系统活跃度
        try:
            # 获取系统活跃度数据
            system_activity = execute_query(
                """SELECT 
                    (SELECT COUNT(DISTINCT mobile) FROM users) as total_users,
                    (SELECT COUNT(*) FROM chat_sessions) as total_sessions,
                    (SELECT COUNT(*) FROM chat_messages WHERE message_type = 'user') as total_questions,
                    (SELECT COUNT(DISTINCT token) FROM chat_sessions WHERE DATE(created_at) >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)) as active_users,
                    (SELECT COUNT(*) FROM content_feedbacks) as feedback_count,
                    (SELECT COUNT(*) FROM content_feedbacks WHERE rating >= 4) as good_rating_count,
                    (SELECT ROUND(COUNT(*) / (SELECT COUNT(*) FROM chat_sessions), 2) FROM chat_messages WHERE message_type = 'user') as avg_messages_per_session
                   """,
                ()
            )
        except Exception as e:
            print(f"获取系统活跃度失败: {str(e)}")
            system_activity = []
        
        # 反馈统计
        try:
            feedback_stats = execute_query(
                """SELECT 
                    (SELECT COUNT(*) FROM content_feedbacks) + (SELECT COUNT(*) FROM user_feedback) as total_feedbacks,
                    (SELECT COUNT(*) FROM user_feedback) as system_feedbacks,
                    (SELECT COUNT(*) FROM content_feedbacks) as content_feedbacks,
                    IFNULL((SELECT AVG(rating) FROM content_feedbacks), 0) as avg_rating
                   FROM dual""",
                ()
            )
        except Exception as e:
            print(f"获取反馈统计失败: {str(e)}")
            feedback_stats = []
        
        # 处理数据格式
        total_users = user_stats[0]['total_users'] if user_stats and len(user_stats) > 0 and 'total_users' in user_stats[0] else 0
        new_users = user_stats[0]['new_users'] if user_stats and len(user_stats) > 0 and 'new_users' in user_stats[0] else 0
        
        knowledge_count = content_stats[0]['knowledge_count'] if content_stats and len(content_stats) > 0 and 'knowledge_count' in content_stats[0] else 0
        safety_count = content_stats[0]['safety_count'] if content_stats and len(content_stats) > 0 and 'safety_count' in content_stats[0] else 0
        emergency_count = content_stats[0]['emergency_count'] if content_stats and len(content_stats) > 0 and 'emergency_count' in content_stats[0] else 0
        
        total_sessions = system_activity[0]['total_sessions'] if system_activity and len(system_activity) > 0 and 'total_sessions' in system_activity[0] else 0
        active_users = system_activity[0]['active_users'] if system_activity and len(system_activity) > 0 and 'active_users' in system_activity[0] else 0
        total_questions = system_activity[0]['total_questions'] if system_activity and len(system_activity) > 0 and 'total_questions' in system_activity[0] else 0
        feedback_count = system_activity[0]['feedback_count'] if system_activity and len(system_activity) > 0 and 'feedback_count' in system_activity[0] else 0
        good_rating_count = system_activity[0]['good_rating_count'] if system_activity and len(system_activity) > 0 and 'good_rating_count' in system_activity[0] else 0
        
        total_feedbacks = feedback_stats[0]['total_feedbacks'] if feedback_stats and len(feedback_stats) > 0 and 'total_feedbacks' in feedback_stats[0] else 0
        system_feedbacks = feedback_stats[0]['system_feedbacks'] if feedback_stats and len(feedback_stats) > 0 and 'system_feedbacks' in feedback_stats[0] else 0
        content_feedbacks = feedback_stats[0]['content_feedbacks'] if feedback_stats and len(feedback_stats) > 0 and 'content_feedbacks' in feedback_stats[0] else 0
        avg_rating = feedback_stats[0]['avg_rating'] if feedback_stats and len(feedback_stats) > 0 and 'avg_rating' in feedback_stats[0] else 0
        
        # 获取前一天数据，计算趋势（实际项目中可能需要更复杂的计算）
        try:
            yesterday_stats = execute_query(
                """SELECT 
                    SUM(CASE WHEN DATE(register_time) = DATE_SUB(CURDATE(), INTERVAL 1 DAY) THEN 1 ELSE 0 END) as yesterday_new_users
                   FROM users""",
                ()
            )
            
            yesterday_new_users = yesterday_stats[0]['yesterday_new_users'] if yesterday_stats and len(yesterday_stats) > 0 and 'yesterday_new_users' in yesterday_stats[0] and yesterday_stats[0]['yesterday_new_users'] else 1
            new_users_trend = ((new_users - yesterday_new_users) / yesterday_new_users) * 100 if yesterday_new_users > 0 else 0
        except Exception as e:
            print(f"计算趋势失败: {str(e)}")
            new_users_trend = 0
        
        # 记录操作日志 (如果提供了管理员ID)
        if admin_id:
            try:
                execute_update(
                    """INSERT INTO operation_logs (admin_id, operation_type, operation_desc, created_at) 
                       VALUES (%s, %s, %s, NOW())""", 
                    (admin_id, "查询", f"管理员{admin_id}访问了仪表盘")
                )
            except Exception as log_error:
                # 仅记录日志错误，不影响主流程
                print(f"记录操作日志失败: {str(log_error)}")
        
        # 返回统计数据
        return {
            "code": 200,
            "message": "获取仪表盘数据成功",
            "data": {
                "user_stats": {
                    "total_users": total_users,
                    "new_users": new_users,
                    "new_users_trend": new_users_trend
                },
                "content_stats": {
                    "knowledge_count": knowledge_count,
                    "safety_count": safety_count,
                    "emergency_count": emergency_count
                },
                "system_activity": {
                    "total_sessions": total_sessions,
                    "active_users": active_users,
                    "total_questions": total_questions,
                    "feedback_count": feedback_count,
                    "good_rating_count": good_rating_count,
                    "avg_messages_per_session": system_activity[0]['avg_messages_per_session'] if system_activity and len(system_activity) > 0 and 'avg_messages_per_session' in system_activity[0] else 0
                },
                "feedback_stats": {
                    "total_feedbacks": total_feedbacks,
                    "system_feedbacks": system_feedbacks,
                    "content_feedbacks": content_feedbacks,
                    "avg_rating": float(avg_rating) if avg_rating else 0
                }
            }
        }
    except Exception as e:
        # 记录错误日志
        import traceback
        traceback_str = traceback.format_exc()
        print(f"获取仪表盘数据失败: {str(e)}")
        print(f"错误详情: {traceback_str}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"获取仪表盘数据失败: {str(e)}")

@router.get("/admin/dashboard/user-activity", tags=["仪表盘"])
async def get_user_activity_chart(
    request: Request,
    days: int = Query(7, description="天数，7或30")
):
    """
    获取用户活跃度趋势图表数据
    """
    try:
        # 从token中获取管理员ID
        try:
            admin_id = await get_admin_id_from_request(request)
        except Exception as auth_error:
            print(f"获取管理员ID失败: {str(auth_error)}")
            admin_id = None
        
        # 验证参数
        if days not in [7, 30]:
            days = 7
        
        try:
            # 构建查询，获取每天的活跃用户数
            activity_data = execute_query(
                """SELECT 
                    DATE(created_at) as date,
                    COUNT(DISTINCT token) as active_users
                   FROM 
                    chat_sessions
                   WHERE 
                    created_at >= DATE_SUB(CURDATE(), INTERVAL %s DAY)
                   GROUP BY 
                    DATE(created_at)
                   ORDER BY 
                    date ASC""",
                (days,)
            )
        except Exception as query_error:
            print(f"查询活跃用户数据失败: {str(query_error)}")
            activity_data = []
        
        # 构建完整的日期范围
        dates = []
        counts = []
        
        current_date = datetime.now().date()
        for i in range(days):
            date = current_date - timedelta(days=days-i-1)
            date_str = date.strftime("%Y-%m-%d")
            dates.append(date_str)
            
            # 查找日期对应的数据
            found = False
            if activity_data:
                for record in activity_data:
                    if 'date' in record and record['date'] and record['date'].strftime("%Y-%m-%d") == date_str:
                        counts.append(record['active_users'] if 'active_users' in record else 0)
                        found = True
                        break
            
            if not found:
                counts.append(0)
        
        # 记录操作日志 (如果提供了管理员ID)
        if admin_id:
            try:
                execute_update(
                    """INSERT INTO operation_logs (admin_id, operation_type, operation_desc, created_at) 
                       VALUES (%s, %s, %s, NOW())""", 
                    (admin_id, "查询", f"管理员{admin_id}查询用户活跃度趋势图数据")
                )
            except Exception as log_error:
                # 仅记录日志错误，不影响主流程
                print(f"记录操作日志失败: {str(log_error)}")
        
        return {
            "code": 200,
            "message": "获取用户活跃度趋势数据成功",
            "data": {
                "dates": dates,
                "counts": counts
            }
        }
    except Exception as e:
        # 记录错误日志
        import traceback
        traceback_str = traceback.format_exc()
        print(f"获取用户活跃度趋势数据失败: {str(e)}")
        print(f"错误详情: {traceback_str}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"获取用户活跃度趋势数据失败: {str(e)}")

@router.get("/admin/dashboard/conversation-trend", tags=["仪表盘"])
async def get_conversation_trend(
    request: Request,
    days: int = Query(7, description="天数，7或30")
):
    """
    获取对话数量趋势图表数据
    """
    try:
        # 从token中获取管理员ID
        try:
            admin_id = await get_admin_id_from_request(request)
        except Exception as auth_error:
            print(f"获取管理员ID失败: {str(auth_error)}")
            admin_id = None
        
        # 验证参数
        if days not in [7, 30]:
            days = 7
        
        try:
            # 构建查询，获取每天的对话数量
            conversation_data = execute_query(
                """SELECT 
                    DATE(created_at) as date,
                    COUNT(*) as conversation_count
                   FROM 
                    chat_sessions
                   WHERE 
                    created_at >= DATE_SUB(CURDATE(), INTERVAL %s DAY)
                   GROUP BY 
                    DATE(created_at)
                   ORDER BY 
                    date ASC""",
                (days,)
            )
        except Exception as query_error:
            print(f"查询对话数量数据失败: {str(query_error)}")
            conversation_data = []
        
        # 构建完整的日期范围
        dates = []
        counts = []
        
        current_date = datetime.now().date()
        for i in range(days):
            date = current_date - timedelta(days=days-i-1)
            date_str = date.strftime("%Y-%m-%d")
            dates.append(date_str)
            
            # 查找日期对应的数据
            found = False
            if conversation_data:
                for record in conversation_data:
                    if 'date' in record and record['date'] and record['date'].strftime("%Y-%m-%d") == date_str:
                        counts.append(record['conversation_count'] if 'conversation_count' in record else 0)
                        found = True
                        break
            
            if not found:
                counts.append(0)
        
        # 记录操作日志 (如果提供了管理员ID)
        if admin_id:
            try:
                execute_update(
                    """INSERT INTO operation_logs (admin_id, operation_type, operation_desc, created_at) 
                       VALUES (%s, %s, %s, NOW())""", 
                    (admin_id, "查询", f"管理员{admin_id}查询对话数量趋势图数据")
                )
            except Exception as log_error:
                # 仅记录日志错误，不影响主流程
                print(f"记录操作日志失败: {str(log_error)}")
        
        return {
            "code": 200,
            "message": "获取对话数量趋势数据成功",
            "data": {
                "dates": dates,
                "counts": counts
            }
        }
    except Exception as e:
        # 记录错误日志
        import traceback
        traceback_str = traceback.format_exc()
        print(f"获取对话数量趋势数据失败: {str(e)}")
        print(f"错误详情: {traceback_str}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"获取对话数量趋势数据失败: {str(e)}")

@router.get("/admin/dashboard/feedback-rating", tags=["仪表盘"])
async def get_feedback_rating_trend(
    request: Request,
    days: int = Query(7, description="天数，7或30")
):
    """
    获取反馈评分趋势图表数据
    """
    try:
        # 从token中获取管理员ID
        try:
            admin_id = await get_admin_id_from_request(request)
        except Exception as auth_error:
            print(f"获取管理员ID失败: {str(auth_error)}")
            admin_id = None
        
        # 验证参数
        if days not in [7, 30]:
            days = 7
        
        try:
            # 构建查询，获取每天的平均评分
            rating_data = execute_query(
                """SELECT 
                    DATE(created_at) as date,
                    AVG(rating) as avg_rating
                   FROM 
                    content_feedbacks
                   WHERE 
                    created_at >= DATE_SUB(CURDATE(), INTERVAL %s DAY)
                   GROUP BY 
                    DATE(created_at)
                   ORDER BY 
                    date ASC""",
                (days,)
            )
        except Exception as query_error:
            print(f"查询评分数据失败: {str(query_error)}")
            rating_data = []
        
        # 构建完整的日期范围
        dates = []
        ratings = []
        
        current_date = datetime.now().date()
        for i in range(days):
            date = current_date - timedelta(days=days-i-1)
            date_str = date.strftime("%Y-%m-%d")
            dates.append(date_str)
            
            # 查找日期对应的数据
            found = False
            if rating_data:
                for record in rating_data:
                    if 'date' in record and record['date'] and record['date'].strftime("%Y-%m-%d") == date_str:
                        ratings.append(float(record['avg_rating']) if 'avg_rating' in record and record['avg_rating'] else 0)
                        found = True
                        break
            
            if not found:
                ratings.append(0)
        
        # 记录操作日志 (如果提供了管理员ID)
        if admin_id:
            try:
                execute_update(
                    """INSERT INTO operation_logs (admin_id, operation_type, operation_desc, created_at) 
                       VALUES (%s, %s, %s, NOW())""", 
                    (admin_id, "查询", f"管理员{admin_id}查询反馈评分趋势图数据")
                )
            except Exception as log_error:
                # 仅记录日志错误，不影响主流程
                print(f"记录操作日志失败: {str(log_error)}")
        
        return {
            "code": 200,
            "message": "获取反馈评分趋势数据成功",
            "data": {
                "dates": dates,
                "ratings": ratings
            }
        }
    except Exception as e:
        # 记录错误日志
        import traceback
        traceback_str = traceback.format_exc()
        print(f"获取反馈评分趋势数据失败: {str(e)}")
        print(f"错误详情: {traceback_str}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"获取反馈评分趋势数据失败: {str(e)}")

@router.get("/admin/dashboard/recent-data", tags=["仪表盘"])
async def get_dashboard_recent_data(request: Request):
    """
    获取仪表盘最近活动数据（最近对话、登录、反馈）
    """
    try:
        # 从token中获取管理员ID
        try:
            admin_id = await get_admin_id_from_request(request)
        except Exception as auth_error:
            print(f"获取管理员ID失败: {str(auth_error)}")
            admin_id = None
            
        # 获取最近对话
        recent_conversations = execute_query(
            """SELECT 
                cs.id, 
                cs.title, 
                cs.token, 
                cs.created_at,
                (SELECT COUNT(*) FROM chat_messages WHERE session_id = cs.id) as message_count,
                u.mobile
               FROM 
                chat_sessions cs
               LEFT JOIN
                users u ON cs.token = u.mobile
               ORDER BY 
                cs.created_at DESC
               LIMIT 5""",
            ()
        )
        
        # 获取最近登录
        recent_logins = execute_query(
            """SELECT 
                ut.user_id, 
                ut.ip_address, 
                ut.created_at as login_time,
                u.mobile
               FROM 
                user_tokens ut
               LEFT JOIN
                users u ON ut.user_id = u.user_id
               WHERE
                ut.is_valid = 1
               ORDER BY 
                ut.created_at DESC
               LIMIT 5""",
            ()
        )
        
        # 如果没有数据，使用管理员登录记录
        if not recent_logins or len(recent_logins) == 0:
            recent_logins = execute_query(
                """SELECT 
                    at.admin_id as user_id, 
                    at.ip_address, 
                    at.created_at as login_time,
                    CONCAT('管理员', at.admin_id) as mobile
                   FROM 
                    admin_tokens at
                   WHERE
                    at.is_valid = 1
                   ORDER BY 
                    at.created_at DESC
                   LIMIT 5""",
                ()
            )
        
        # 获取最新反馈 - 修复字符集冲突问题
        # 分别查询两个表并在应用层合并，而不是使用UNION
        content_feedbacks = execute_query(
            """SELECT 
                'content' as feedback_type,
                rating,
                feedback as content,
                feedback_option as type,
                created_at
               FROM 
                content_feedbacks
               ORDER BY 
                created_at DESC
               LIMIT 3""",
            ()
        )
        
        system_feedbacks = execute_query(
            """SELECT 
                'system' as feedback_type,
                NULL as rating,
                feedback_content as content,
                feedback_type as type,
                created_at
               FROM 
                user_feedback
               ORDER BY 
                created_at DESC
               LIMIT 3""",
            ()
        )
        
        # 在应用层合并并排序
        recent_feedbacks = []
        if content_feedbacks:
            recent_feedbacks.extend(content_feedbacks)
        if system_feedbacks:
            recent_feedbacks.extend(system_feedbacks)
        
        # 根据created_at排序，取最近5条
        recent_feedbacks.sort(key=lambda x: x.get('created_at', datetime.now()), reverse=True)
        recent_feedbacks = recent_feedbacks[:5]
        
        # 处理数据格式
        for conv in recent_conversations:
            if 'created_at' in conv and conv['created_at']:
                conv['created_at'] = conv['created_at'].strftime("%Y-%m-%d %H:%M:%S")
            else:
                conv['created_at'] = ""
        
        for login in recent_logins:
            if 'login_time' in login and login['login_time']:
                login['login_time'] = login['login_time'].strftime("%Y-%m-%d %H:%M:%S")
            else:
                login['login_time'] = ""
        
        for feedback in recent_feedbacks:
            if 'created_at' in feedback and feedback['created_at']:
                feedback['created_at'] = feedback['created_at'].strftime("%Y-%m-%d %H:%M:%S")
            else:
                feedback['created_at'] = ""
                
            # 转换反馈类型为前端需要的格式
            if 'type' in feedback and feedback['type'] in ['positive', 'negative', 'suggestion']:
                feedback['typeLabel'] = {
                    'positive': '正面',
                    'negative': '负面',
                    'suggestion': '建议'
                }.get(feedback['type'], '其他')
            else:
                feedback['typeLabel'] = feedback.get('type', '其他')
        
        # 记录操作日志 (如果提供了管理员ID)
        if admin_id:
            try:
                execute_update(
                    """INSERT INTO operation_logs (admin_id, operation_type, operation_desc, created_at) 
                       VALUES (%s, %s, %s, NOW())""", 
                    (admin_id, "查询", f"管理员{admin_id}查询仪表盘最近活动数据")
                )
            except Exception as log_error:
                # 仅记录日志错误，不影响主流程
                print(f"记录操作日志失败: {str(log_error)}")
        
        # 如果某个数据为空，确保返回空列表而不是None
        return {
            "code": 200,
            "message": "获取最近活动数据成功",
            "data": {
                "recent_conversations": recent_conversations or [],
                "recent_logins": recent_logins or [],
                "recent_feedbacks": recent_feedbacks or []
            }
        }
    except Exception as e:
        # 记录详细错误日志
        import traceback
        traceback_str = traceback.format_exc()
        print(f"获取最近活动数据失败: {str(e)}")
        print(f"错误详情: {traceback_str}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"获取最近活动数据失败: {str(e)}")

@router.get("/admin/dashboard/system-info", tags=["仪表盘"])
async def get_dashboard_system_info(request: Request):
    """
    获取仪表盘系统信息（系统参数、操作日志、系统版本）
    """
    try:
        # 从token中获取管理员ID
        try:
            admin_id = await get_admin_id_from_request(request)
        except Exception as auth_error:
            print(f"获取管理员ID失败: {str(auth_error)}")
            admin_id = None
        
        # 获取系统参数
        system_params = execute_query(
            """SELECT 
                config_key as `key`, 
                config_value as value,
                description
               FROM 
                system_configs
               ORDER BY 
                config_id ASC
               LIMIT 4""",
            ()
        )
        
        # 转换为前端需要的格式
        system_params_list = []
        if system_params:
            for param in system_params:
                if param and 'description' in param and 'key' in param and 'value' in param:
                    name = param['description'] if param['description'] else param['key']
                    system_params_list.append({
                        'name': name,
                        'value': param['value']
                    })
        
        # 获取最近操作日志
        operation_logs = execute_query(
            """SELECT 
                ol.admin_id,
                IFNULL(a.full_name, CONCAT('管理员', ol.admin_id)) as admin_name,
                ol.operation_type,
                ol.operation_desc as action,
                ol.created_at as operation_time
               FROM 
                operation_logs ol
               LEFT JOIN
                admins a ON ol.admin_id = a.admin_id
               ORDER BY 
                ol.created_at DESC
               LIMIT 4""",
            ()
        )
        
        # 处理操作时间格式
        if operation_logs:
            for log in operation_logs:
                if log and 'operation_time' in log:
                    try:
                        # 计算相对时间（如"10分钟前"）
                        time_diff = datetime.now() - log['operation_time']
                        minutes = time_diff.total_seconds() / 60
                        
                        if minutes < 1:
                            log['operation_time'] = "刚刚"
                        elif minutes < 60:
                            log['operation_time'] = f"{int(minutes)}分钟前"
                        elif minutes < 1440:  # 24小时内
                            log['operation_time'] = f"{int(minutes/60)}小时前"
                        else:
                            log['operation_time'] = log['operation_time'].strftime("%Y-%m-%d %H:%M")
                    except Exception as time_error:
                        print(f"处理操作时间格式失败: {str(time_error)}")
                        log['operation_time'] = str(log['operation_time'])
        
        # 获取系统版本
        system_version = execute_query(
            """SELECT 
                version_number,
                knowledge_base_version,
                release_date,
                update_notes
               FROM 
                system_versions
               WHERE 
                is_current = 1
               LIMIT 1""",
            ()
        )
        
        version_info = {}
        if system_version and len(system_version) > 0:
            version_info = system_version[0]
            # 格式化日期
            if 'release_date' in version_info and version_info['release_date'] and not isinstance(version_info['release_date'], str):
                try:
                    version_info['release_date'] = version_info['release_date'].strftime("%Y-%m-%d")
                except Exception as date_error:
                    print(f"格式化日期失败: {str(date_error)}")
                    version_info['release_date'] = str(version_info['release_date'])
        else:
            version_info = {
                'version_number': '未知版本',
                'knowledge_base_version': '未知版本',
                'release_date': '未知日期',
                'update_notes': '暂无更新说明'
            }
        
        # 获取待处理事项数量
        try:
            pending_items = execute_query(
                """SELECT 
                    (SELECT COUNT(*) FROM user_feedback WHERE status = 'pending') as pending_feedbacks,
                    (SELECT COUNT(*) FROM knowledge_documents WHERE is_published = 0) as pending_documents,
                    0 as system_warnings
                   FROM dual""",
                ()
            )
            
            pending_counts = pending_items[0] if pending_items and len(pending_items) > 0 else {
                'pending_feedbacks': 0,
                'pending_documents': 0,
                'system_warnings': 0
            }
        except Exception as pending_error:
            print(f"获取待处理事项数量失败: {str(pending_error)}")
            pending_counts = {
                'pending_feedbacks': 0,
                'pending_documents': 0,
                'system_warnings': 0
            }
        
        # 记录操作日志 (如果提供了管理员ID)
        if admin_id:
            try:
                execute_update(
                    """INSERT INTO operation_logs (admin_id, operation_type, operation_desc, created_at) 
                       VALUES (%s, %s, %s, NOW())""", 
                    (admin_id, "查询", f"管理员{admin_id}查询仪表盘系统信息")
                )
            except Exception as log_error:
                # 仅记录日志错误，不影响主流程
                print(f"记录操作日志失败: {str(log_error)}")
        
        return {
            "code": 200,
            "message": "获取系统信息成功",
            "data": {
                "system_params": system_params_list,
                "operation_logs": operation_logs or [],
                "system_version": version_info,
                "pending_items": {
                    "feedbacks": pending_counts.get('pending_feedbacks', 0),
                    "documents": pending_counts.get('pending_documents', 0),
                    "warnings": pending_counts.get('system_warnings', 0)
                }
            }
        }
    except Exception as e:
        # 记录错误日志
        import traceback
        traceback_str = traceback.format_exc()
        print(f"获取系统信息失败: {str(e)}")
        print(f"错误详情: {traceback_str}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"获取系统信息失败: {str(e)}")
