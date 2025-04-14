# 引入 FastAPI 中的 APIRouter 和 HTTPException 模块，用于创建路由和处理异常
from fastapi import APIRouter, HTTPException, Query, Request, Depends
# 引入 Pydantic 中的 BaseModel 类，用于定义请求体的数据结构和验证
from pydantic import BaseModel
# 从数据库模块导入 execute_query 和 execute_update 函数，用于执行查询和更新操作
from database import execute_query, execute_update
# 引入时间模块处理日期范围
from datetime import datetime, timedelta
# 引入 typing 模块中的 Optional 和 List 类型
from typing import Optional, List
import traceback
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化 APIRouter 实例，用于定义路由
router = APIRouter()

# 定义请求体的模型，用于更新反馈处理状态
class UpdateFeedbackStatusRequest(BaseModel):
    status: str  # 反馈状态：pending, processing, resolved, rejected
    admin_reply: Optional[str] = None  # 管理员回复内容

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
            raise HTTPException(status_code=401, detail="无效的token或token已过期")
        
        admin_id = admin_result[0]['admin_id']
        
        # 验证管理员是否存在
        admin_info = execute_query(
            """SELECT admin_id, full_name FROM admins WHERE admin_id = %s""",
            (admin_id,)
        )
        
        if not admin_info:
            raise HTTPException(status_code=401, detail="管理员不存在")
        
        return admin_id
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"管理员认证失败: {str(e)}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        raise HTTPException(status_code=401, detail=f"管理员认证失败: {str(e)}")

# 安全记录管理员操作
def log_admin_operation(admin_id: int, operation_type: str, description: str):
    if not admin_id:
        return
    
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
            """INSERT INTO operation_logs (admin_id, operation_type, operation_desc, created_at) 
               VALUES (%s, %s, %s, NOW())""",
            (admin_id, operation_type, full_description)
        )
    except Exception as e:
        # 记录错误但不中断主要流程
        logger.error(f"记录操作日志失败: {str(e)}")

# 检查content_feedbacks表是否存在status列
def check_and_update_content_feedbacks_table():
    try:
        # 检查表结构
        check_query = """
            SELECT COLUMN_NAME 
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA = 'chemical_server' 
            AND TABLE_NAME = 'content_feedbacks'
            AND COLUMN_NAME = 'status'
        """
        result = execute_query(check_query)
        
        # 如果status列不存在，添加它
        if not result:
            alter_query = """
                ALTER TABLE content_feedbacks 
                ADD COLUMN status ENUM('pending', 'processing', 'resolved', 'rejected') 
                NOT NULL DEFAULT 'pending',
                ADD COLUMN admin_reply TEXT NULL,
                ADD COLUMN replied_at TIMESTAMP NULL
            """
            execute_update(alter_query)
            # 保留重要信息而不是简单调试
            logger.info("数据库更新: content_feedbacks表已更新，添加了status, admin_reply和replied_at列")
        
        # 更新所有没有状态的记录为pending
        update_query = """
            UPDATE content_feedbacks 
            SET status = 'pending' 
            WHERE status IS NULL OR status = ''
        """
        execute_update(update_query)
        
    except Exception as e:
        logger.error(f"检查/更新content_feedbacks表结构失败: {str(e)}")

# 应用启动时检查表结构
check_and_update_content_feedbacks_table()

# 获取AI内容反馈列表接口
@router.get("/admin/feedback/content", tags=["反馈管理"])
async def get_content_feedback_list(
    request: Request,
    page: int = Query(1, description="页码"),
    page_size: int = Query(10, description="每页数量"),
    rating: Optional[int] = Query(None, description="评分筛选"),
    feedback_option: Optional[str] = Query(None, description="反馈选项筛选"),
    status: Optional[str] = Query(None, description="处理状态筛选"),
    start_date: Optional[str] = Query(None, description="开始日期"),
    end_date: Optional[str] = Query(None, description="结束日期"),
    keyword: Optional[str] = Query(None, description="关键词搜索")
):
    """
    获取AI内容反馈列表，支持分页和筛选
    从content_feedbacks表获取数据
    """
    try:
        # 获取管理员ID
        admin_id = await get_current_admin(request)
        
        # 计算偏移量
        offset = (page - 1) * page_size
        
        # 构建基础查询SQL
        query = """
            SELECT 
                id as feedback_id, rating, feedback, feedback_option, 
                message, question, created_at, 
                COALESCE(status, 'pending') as status, 
                admin_reply, replied_at
            FROM 
                content_feedbacks
            WHERE 1=1
        """
        params = []
        
        # 添加筛选条件
        if rating is not None:  # 确保只在rating非None时添加条件
            try:
                if isinstance(rating, str) and rating.strip():  # 如果是非空字符串
                    rating = int(rating)  # 尝试转换为整数
                if isinstance(rating, int):  # 确保是整数类型
                    query += " AND rating = %s"
                    params.append(rating)
            except (ValueError, TypeError) as e:
                logger.error(f"评分筛选参数转换错误: {str(e)}")
                # 不添加此筛选条件
        
        if feedback_option:
            query += " AND feedback_option = %s"
            params.append(feedback_option)
        
        if status:
            query += " AND COALESCE(status, 'pending') = %s"
            params.append(status)
        
        if start_date:
            query += " AND DATE(created_at) >= %s"
            params.append(start_date)
        
        if end_date:
            query += " AND DATE(created_at) <= %s"
            params.append(end_date)
        
        if keyword:
            query += " AND (feedback LIKE %s OR message LIKE %s OR question LIKE %s)"
            keyword_param = f"%{keyword}%"
            params.extend([keyword_param, keyword_param, keyword_param])
        
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
        
        # 处理日期时间格式
        for feedback in feedback_list:
            feedback['created_at'] = feedback['created_at'].strftime("%Y-%m-%d %H:%M:%S") if feedback['created_at'] else None
            feedback['replied_at'] = feedback['replied_at'].strftime("%Y-%m-%d %H:%M:%S") if feedback['replied_at'] else None
            # 确保status字段有值
            if not feedback.get('status'):
                feedback['status'] = 'pending'
        
        # 记录操作日志
        log_admin_operation(admin_id, "查询", "查询AI内容反馈列表")
        
        return {
            "code": 200,
            "message": "获取AI内容反馈列表成功",
            "data": {
                "list": feedback_list,
                "total": total_count
            }
        }
    except Exception as e:
        # 记录错误日志
        logger.error(f"获取AI内容反馈列表失败: {str(e)}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        logger.error(f"SQL查询: {query if 'query' in locals() else '未构建查询'}")
        logger.error(f"参数: {params if 'params' in locals() else '未构建参数'}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"获取AI内容反馈列表失败: {str(e)}")

# 获取AI内容反馈状态统计接口
@router.get("/admin/feedback/content/stats", tags=["反馈管理"])
async def get_content_feedback_stats(
    request: Request,
    rating: Optional[int] = Query(None, description="评分筛选"),
    feedback_option: Optional[str] = Query(None, description="反馈选项筛选"),
    start_date: Optional[str] = Query(None, description="开始日期"),
    end_date: Optional[str] = Query(None, description="结束日期"),
    keyword: Optional[str] = Query(None, description="关键词搜索")
):
    """
    获取AI内容反馈状态统计
    """
    try:
        # 获取管理员ID
        admin_id = await get_current_admin(request)
        
        # 构建基础条件
        condition = "WHERE 1=1"
        params = []
        
        # 添加筛选条件
        if rating:
            condition += " AND rating = %s"
            params.append(rating)
        
        if feedback_option:
            condition += " AND feedback_option = %s"
            params.append(feedback_option)
        
        if start_date:
            condition += " AND DATE(created_at) >= %s"
            params.append(start_date)
        
        if end_date:
            condition += " AND DATE(created_at) <= %s"
            params.append(end_date)
        
        if keyword:
            condition += " AND (feedback LIKE %s OR message LIKE %s OR question LIKE %s)"
            keyword_param = f"%{keyword}%"
            params.extend([keyword_param, keyword_param, keyword_param])
        
        # 状态统计查询
        status_query = f"""
            SELECT 
                COALESCE(status, 'pending') as status, COUNT(*) as count
            FROM 
                content_feedbacks
            {condition}
            GROUP BY COALESCE(status, 'pending')
        """
        
        # 执行状态统计查询
        status_counts = execute_query(status_query, tuple(params))
        
        # 处理结果为字典格式
        result_status = {}
        for item in status_counts:
            result_status[item['status']] = item['count']
        
        # 确保所有状态都有值，即使是0
        for status in ['pending', 'processing', 'resolved', 'rejected']:
            if status not in result_status:
                result_status[status] = 0
        
        # 总体统计查询
        overview_query = f"""
            SELECT 
                COUNT(*) as total,
                AVG(rating) as avg_score
            FROM 
                content_feedbacks
            {condition}
        """
        
        # 执行总体统计查询
        overview_result = execute_query(overview_query, tuple(params))
        overview_data = overview_result[0] if overview_result else {
            'total': 0,
            'avg_score': 0
        }
        
        # 记录操作日志
        log_admin_operation(admin_id, "查询", "查询AI内容反馈统计")
        
        return {
            "code": 200,
            "message": "获取AI内容反馈状态统计成功",
            "data": {
                "status_counts": result_status,
                "overview": overview_data
            }
        }
    except Exception as e:
        # 记录错误日志
        logger.error(f"获取AI内容反馈状态统计失败: {str(e)}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"获取AI内容反馈状态统计失败: {str(e)}")

# 获取AI内容反馈详情接口
@router.get("/admin/feedback/content/{feedback_id}", tags=["反馈管理"])
async def get_content_feedback_detail(
    request: Request,
    feedback_id: int
):
    """
    获取AI内容反馈详情
    """
    try:
        # 获取管理员ID
        admin_id = await get_current_admin(request)
        
        # 查询反馈详情
        query = """
            SELECT 
                id as feedback_id, rating, feedback, feedback_option, 
                message, question, created_at, 
                COALESCE(status, 'pending') as status, 
                admin_reply, replied_at
            FROM 
                content_feedbacks
            WHERE 
                id = %s
            LIMIT 1
        """
        feedback_detail = execute_query(query, (feedback_id,))
        
        if not feedback_detail:
            raise HTTPException(status_code=404, detail=f"未找到ID为{feedback_id}的反馈记录")
        
        # 处理日期时间格式
        feedback = feedback_detail[0]
        feedback['created_at'] = feedback['created_at'].strftime("%Y-%m-%d %H:%M:%S") if feedback['created_at'] else None
        feedback['replied_at'] = feedback['replied_at'].strftime("%Y-%m-%d %H:%M:%S") if feedback['replied_at'] else None
        
        # 记录操作日志
        log_admin_operation(admin_id, "查询", f"查看AI内容反馈详情[ID:{feedback_id}]")
        
        return {
            "code": 200,
            "message": "获取AI内容反馈详情成功",
            "data": feedback
        }
    except HTTPException:
        raise
    except Exception as e:
        # 记录错误日志
        logger.error(f"获取AI内容反馈详情失败: {str(e)}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"获取AI内容反馈详情失败: {str(e)}")

# 更新AI内容反馈状态接口
@router.put("/admin/feedback/content/{feedback_id}", tags=["反馈管理"])
async def update_content_feedback_status(
    request: Request,
    feedback_id: int,
    update_request: UpdateFeedbackStatusRequest
):
    """
    更新AI内容反馈状态
    """
    try:
        # 获取管理员ID
        admin_id = await get_current_admin(request)
        
        # 验证状态值有效性
        valid_statuses = ['pending', 'processing', 'resolved', 'rejected']
        if update_request.status not in valid_statuses:
            raise HTTPException(status_code=400, detail=f"无效的状态值: {update_request.status}")
        
        # 查询当前反馈状态
        check_query = "SELECT COALESCE(status, 'pending') as status FROM content_feedbacks WHERE id = %s LIMIT 1"
        check_result = execute_query(check_query, (feedback_id,))
        
        if not check_result:
            raise HTTPException(status_code=404, detail=f"未找到ID为{feedback_id}的反馈记录")
        
        # 如果已经是已解决或已拒绝状态，且尝试再次更新为这些状态，则不做任何更改
        current_status = check_result[0]['status']
        if current_status in ['resolved', 'rejected'] and current_status == update_request.status:
            return {
                "code": 200,
                "message": f"反馈已经是{update_request.status}状态，无需更新",
                "data": None
            }
        
        # 更新反馈状态
        update_query = """
            UPDATE content_feedbacks 
            SET status = %s, admin_reply = %s, replied_at = NOW() 
            WHERE id = %s
        """
        execute_update(update_query, (update_request.status, update_request.admin_reply, feedback_id))
        
        # 记录操作日志
        log_admin_operation(admin_id, "更新", f"更新AI内容反馈[ID:{feedback_id}]状态为{update_request.status}")
        
        return {
            "code": 200,
            "message": "更新反馈状态成功",
            "data": None
        }
    except HTTPException:
        raise
    except Exception as e:
        # 记录错误日志
        logger.error(f"更新反馈状态失败: {str(e)}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"更新反馈状态失败: {str(e)}")
