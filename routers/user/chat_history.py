# 引入 FastAPI 中的 APIRouter 和 HTTPException 模块，用于创建路由和处理异常
from fastapi import APIRouter, HTTPException, Depends, Request
# 引入 Pydantic 中的 BaseModel 类，用于定义请求体的数据结构和验证
from pydantic import BaseModel, Field
# 从数据库模块导入 execute_query 和 execute_update 函数，用于执行查询和更新操作
from database import execute_query, execute_update
# 引入 UUID 模块用于生成唯一标识符
import uuid
# 引入可选类型和列表类型
from typing import Optional, List, Dict, Any, Union
# 引入时间模块
from datetime import datetime
# 引入 json 模块处理 JSON 数据
import json
import logging

# 初始化日志记录器
logger = logging.getLogger(__name__)

# 初始化 APIRouter 实例，用于定义路由
router = APIRouter()

# 定义请求和响应模型
class ChatSessionCreate(BaseModel):
    """创建聊天会话的请求模型"""
    title: Optional[str] = None

class ChatMessageCreate(BaseModel):
    """创建聊天消息的请求模型"""
    id: str
    session_id: str  
    message_type: str  # 消息类型: "user" 或 "ai"
    content: Optional[str] = None  # 消息内容
    parent_id: Optional[str] = None  # 父消息ID，回复时使用
    paired_ai_id: Optional[str] = None  # 配对的AI消息ID，用户消息使用
    message_references: Optional[str] = Field(default='{}')  # 消息引用的内容，作为JSON字符串
    question: Optional[str] = None  # 相关问题
    is_loading: Optional[bool] = False  # 是否处于加载状态

class ChatMessageUpdate(BaseModel):
    """更新聊天消息的请求模型"""
    content: Optional[str] = None
    message_references: Optional[str] = None  # 消息引用的内容，作为JSON字符串
    is_loading: Optional[bool] = None

class ChatSessionUpdate(BaseModel):
    """更新聊天会话的请求模型"""
    title: Optional[str] = None

# 从请求中获取token
async def get_token_from_request(request: Request) -> str:
    """
    从Authorization头中提取token
    """
    try:
        auth_header = request.headers.get('Authorization')
        
        if not auth_header or not auth_header.startswith('Bearer '):
            raise HTTPException(status_code=401, detail="无效的认证信息")
        
        token = auth_header.split(' ')[1]
        
        # 验证token是否有效（admin_tokens或user_tokens中存在且有效）
        admin_result = execute_query(
            "SELECT admin_id FROM admin_tokens WHERE token = %s AND is_valid = 1 AND expire_at > NOW()",
            (token,)
        )
        
        if admin_result:
            return token
        
        user_result = execute_query(
            "SELECT user_id FROM user_tokens WHERE token = %s AND is_valid = 1 AND expire_at > NOW()",
            (token,)
        )
        
        if user_result:
            return token
        
        raise HTTPException(status_code=401, detail="无效的token")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"认证失败: {str(e)}")

# API 路由

@router.post("/chat/sessions", tags=["聊天历史"])
async def create_chat_session(request: Request, session: ChatSessionCreate):
    """
    创建新的聊天会话。
    """
    try:
        token = await get_token_from_request(request)
        session_id = str(uuid.uuid4())
        title = session.title or f"对话 {datetime.now().strftime('%H:%M:%S')}"
        
        execute_update(
            """INSERT INTO chat_sessions (id, token, title, created_at) 
               VALUES (%s, %s, %s, NOW())""", 
            (session_id, token, title)
        )
        
        return {"id": session_id, "title": title}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建会话失败: {str(e)}")

@router.get("/chat/sessions", tags=["聊天历史"])
async def get_user_chat_sessions(request: Request):
    """
    获取用户的所有聊天会话。
    """
    try:
        token = await get_token_from_request(request)
        
        sessions = execute_query(
            """SELECT id, title, created_at, updated_at 
               FROM chat_sessions 
               WHERE token = %s 
               ORDER BY updated_at DESC""", 
            (token,)
        )
        
        for session in sessions:
            session['created_at'] = session['created_at'].isoformat() if session['created_at'] else None
            session['updated_at'] = session['updated_at'].isoformat() if session['updated_at'] else None
        
        return {"sessions": sessions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取会话列表失败: {str(e)}")

@router.get("/chat/sessions/{session_id}/messages", tags=["聊天历史"])
async def get_chat_session_messages(request: Request, session_id: str):
    """
    获取特定聊天会话的所有消息。
    """
    try:
        token = await get_token_from_request(request)
        
        session_check = execute_query(
            """SELECT id FROM chat_sessions WHERE id = %s AND token = %s""", 
            (session_id, token)
        )
        
        if not session_check:
            raise HTTPException(status_code=403, detail="无权访问该会话")
        
        messages = execute_query(
            """SELECT id, message_type, content, parent_id, paired_ai_id, 
                      message_references, question, is_loading, created_at
               FROM chat_messages 
               WHERE session_id = %s 
               ORDER BY created_at ASC""", 
            (session_id,)
        )
        
        for message in messages:
            message['created_at'] = message['created_at'].isoformat() if message['created_at'] else None
            message['is_loading'] = bool(message['is_loading'])
            
            if isinstance(message['message_references'], str):
                try:
                    message['message_references'] = json.loads(message['message_references'])
                except json.JSONDecodeError:
                    pass
        
        return {"messages": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取消息列表失败: {str(e)}")

@router.post("/chat/messages", tags=["聊天历史"])
async def create_chat_message(request: Request, message: ChatMessageCreate):
    """
    创建新的聊天消息。
    """
    try:
        token = await get_token_from_request(request)
        
        session_check = execute_query(
            """SELECT id FROM chat_sessions WHERE id = %s AND token = %s""", 
            (message.session_id, token)
        )
        
        if not session_check:
            raise HTTPException(status_code=403, detail="无权访问该会话")
            
        message_references = message.message_references or '{}'
        try:
            json.loads(message_references)
        except json.JSONDecodeError:
            message_references = '{}'
        
        execute_update(
            """INSERT INTO chat_messages 
               (id, session_id, message_type, content, parent_id, paired_ai_id, 
                message_references, question, is_loading, created_at) 
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())""", 
            (
                message.id, 
                message.session_id, 
                message.message_type, 
                message.content,
                message.parent_id, 
                message.paired_ai_id,
                message_references,
                message.question,
                message.is_loading
            )
        )
        
        execute_update(
            """UPDATE chat_sessions SET updated_at = NOW() WHERE id = %s""", 
            (message.session_id,)
        )
        
        return {"id": message.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建消息失败: {str(e)}")

@router.put("/chat/messages/{message_id}", tags=["聊天历史"])
async def update_chat_message(request: Request, message_id: str, message: ChatMessageUpdate):
    """
    更新现有的聊天消息。
    """
    try:
        token = await get_token_from_request(request)
        
        message_info = execute_query(
            """SELECT session_id FROM chat_messages WHERE id = %s""",
            (message_id,)
        )
        
        if not message_info:
            raise HTTPException(status_code=404, detail="消息不存在")
            
        session_id = message_info[0]['session_id']
        
        session_check = execute_query(
            """SELECT id FROM chat_sessions WHERE id = %s AND token = %s""", 
            (session_id, token)
        )
        
        if not session_check:
            raise HTTPException(status_code=403, detail="无权修改该消息")
        
        update_fields = []
        params = []
        
        if message.content is not None:
            update_fields.append("content = %s")
            params.append(message.content)
            
        if message.message_references is not None:
            update_fields.append("message_references = %s")
            try:
                json.loads(message.message_references)
                message_references = message.message_references
            except json.JSONDecodeError:
                message_references = '{}'
            params.append(message_references)
            
        if message.is_loading is not None:
            update_fields.append("is_loading = %s")
            params.append(message.is_loading)
        
        if not update_fields:
            return {"message": "Nothing to update"}
        
        params.append(message_id)
        
        execute_update(
            f"""UPDATE chat_messages 
                SET {", ".join(update_fields)} 
                WHERE id = %s""", 
            tuple(params)
        )
        
        return {"message": "Message updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新消息失败: {str(e)}")

@router.put("/chat/sessions/{session_id}", tags=["聊天历史"])
async def update_chat_session(request: Request, session_id: str, session: ChatSessionUpdate):
    """
    更新聊天会话信息，例如标题。
    """
    try:
        token = await get_token_from_request(request)
        
        session_check = execute_query(
            """SELECT id FROM chat_sessions WHERE id = %s AND token = %s""", 
            (session_id, token)
        )
        
        if not session_check:
            raise HTTPException(status_code=403, detail="无权修改该会话")
            
        if session.title:
            execute_update(
                """UPDATE chat_sessions SET title = %s, updated_at = NOW() WHERE id = %s""", 
                (session.title, session_id)
            )
            
        return {"message": "Session updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新会话失败: {str(e)}")

@router.delete("/chat/sessions/{session_id}", tags=["聊天历史"])
async def delete_chat_session(request: Request, session_id: str):
    """
    删除聊天会话及其所有消息。
    """
    try:
        token = await get_token_from_request(request)
        
        session_check = execute_query(
            """SELECT id FROM chat_sessions WHERE id = %s AND token = %s""", 
            (session_id, token)
        )
        
        if not session_check:
            raise HTTPException(status_code=403, detail="无权删除该会话")
            
        result = execute_update(
            """DELETE FROM chat_sessions WHERE id = %s AND token = %s""", 
            (session_id, token)
        )
        
        if result == 0:
            raise HTTPException(status_code=404, detail="Chat session not found")
            
        return {"message": "Session deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        # 记录错误
        logger.error(f"删除会话 {session_id} 失败: {str(e)}")
        return {"code": 500, "message": f"删除会话失败: {str(e)}"}

@router.delete("/chat/sessions", tags=["聊天历史"])
async def delete_all_user_chat_sessions(request: Request):
    """
    删除用户的所有聊天会话及其消息。
    """
    try:
        token = await get_token_from_request(request)
        
        # 验证token并获取用户ID
        user_result = execute_query(
            "SELECT user_id FROM user_tokens WHERE token = %s AND is_valid = 1 AND expire_at > NOW()",
            (token,)
        )
        
        if not user_result:
            raise HTTPException(status_code=401, detail="无效的token")
        
        user_id = user_result[0]['user_id']
        
        # 获取用户的所有有效token
        user_tokens = execute_query(
            """SELECT token FROM user_tokens WHERE user_id = %s AND is_valid = 1""",
            (user_id,)
        )
        
        if not user_tokens:
            return {"message": "没有找到可删除的会话"}
        
        token_list = [t['token'] for t in user_tokens]
        
        # 使用JOIN查询获取所有相关会话
        sessions = execute_query(
            """
            SELECT DISTINCT cs.id 
            FROM chat_sessions cs
            JOIN user_tokens ut ON cs.token = ut.token
            WHERE ut.user_id = %s AND ut.is_valid = 1
            """,
            (user_id,)
        )
        
        if not sessions:
            return {"message": "没有找到可删除的会话"}
        
        session_ids = [s['id'] for s in sessions]
        delete_count = 0
        
        for session_id in session_ids:
            try:
                # 先删除消息
                execute_update(
                    """DELETE FROM chat_messages WHERE session_id = %s""", 
                    (session_id,)
                )
                
                # 再删除会话
                session_result = execute_update(
                    """DELETE FROM chat_sessions WHERE id = %s""", 
                    (session_id,)
                )
                
                if session_result > 0:
                    delete_count += 1
            except Exception as delete_error:
                # 记录错误
                logger.error(f"删除会话 {session_id} 失败: {str(delete_error)}")
        
        return {"message": f"已成功删除 {delete_count} 个会话"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除会话失败: {str(e)}")
