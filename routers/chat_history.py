from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from database import execute_query, execute_update
from typing import Optional, List, Dict, Any
import uuid
from datetime import datetime
from routers.login import get_current_user

# 初始化路由
router = APIRouter()
security = HTTPBearer()

# 定义请求和响应模型
class ChatSessionCreate(BaseModel):
    user_id: str
    title: Optional[str] = None

class ChatMessageCreate(BaseModel):
    session_id: str
    message_type: str  # 'user' 或 'ai'
    content: str
    parent_id: Optional[str] = None
    paired_ai_id: Optional[str] = None
    references: Optional[List[Dict[str, Any]]] = None
    question: Optional[str] = None
    is_loading: Optional[bool] = False

# 创建新会话
@router.post("/chat/sessions")
async def create_chat_session(request: ChatSessionCreate):
    try:
        # 检查用户是否存在
        user_result = execute_query("SELECT * FROM users WHERE user_id = %s", (request.user_id,))
        if not user_result:
            return {"code": 404, "message": "用户不存在"}
        
        # 生成会话ID
        session_id = str(uuid.uuid4())
        
        # 生成默认标题（如果未提供）
        title = request.title or f"对话 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # 创建新会话
        execute_update(
            "INSERT INTO chat_sessions (id, user_id, title, created_at) VALUES (%s, %s, %s, NOW())",
            (session_id, request.user_id, title)
        )
        
        return {"id": session_id, "title": title}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 获取用户的所有会话
@router.get("/chat/sessions")
async def get_chat_sessions(user_id: str = Query(...)):
    try:
        # 查询用户的所有会话
        sessions = execute_query(
            """SELECT id, user_id, title, created_at, updated_at 
               FROM chat_sessions 
               WHERE user_id = %s 
               ORDER BY updated_at DESC""",
            (user_id,)
        )
        
        return sessions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 获取会话详情
@router.get("/chat/sessions/{session_id}")
async def get_chat_session(session_id: str):
    try:
        # 查询会话信息
        session = execute_query(
            "SELECT id, user_id, title, created_at, updated_at FROM chat_sessions WHERE id = %s",
            (session_id,)
        )
        
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")
        
        return session[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 删除会话
@router.delete("/chat/sessions/{session_id}")
async def delete_chat_session(session_id: str):
    try:
        # 检查会话是否存在
        session = execute_query("SELECT * FROM chat_sessions WHERE id = %s", (session_id,))
        
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")
        
        # 删除会话（会级联删除该会话下的所有消息）
        execute_update("DELETE FROM chat_sessions WHERE id = %s", (session_id,))
        
        return {"code": 200, "message": "会话已删除"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 清空用户的所有会话
@router.delete("/chat/sessions/clear")
async def clear_chat_sessions(user_id: str = Query(...)):
    try:
        # 检查用户是否存在
        user_result = execute_query("SELECT * FROM users WHERE user_id = %s", (user_id,))
        
        if not user_result:
            raise HTTPException(status_code=404, detail="用户不存在")
        
        # 删除用户的所有会话（会级联删除所有消息）
        execute_update("DELETE FROM chat_sessions WHERE user_id = %s", (user_id,))
        
        return {"code": 200, "message": "所有会话已清空"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 获取会话中的所有消息
@router.get("/chat/messages")
async def get_chat_messages(session_id: str = Query(...)):
    try:
        # 检查会话是否存在
        session = execute_query("SELECT * FROM chat_sessions WHERE id = %s", (session_id,))
        
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")
        
        # 查询会话中的所有消息
        messages = execute_query(
            """SELECT id, session_id, message_type, content, parent_id, paired_ai_id, 
               references, question, is_loading, created_at 
               FROM chat_messages 
               WHERE session_id = %s 
               ORDER BY created_at ASC""",
            (session_id,)
        )
        
        return messages
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 创建新消息
@router.post("/chat/messages")
async def create_chat_message(request: ChatMessageCreate):
    try:
        # 检查会话是否存在
        session = execute_query("SELECT * FROM chat_sessions WHERE id = %s", (request.session_id,))
        
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")
        
        # 生成消息ID
        message_id = str(uuid.uuid4())
        
        # 处理可能为None的字段
        parent_id = request.parent_id if request.parent_id is not None else None
        paired_ai_id = request.paired_ai_id if request.paired_ai_id is not None else None
        references = request.references if request.references is not None else []
        question = request.question if request.question is not None else ""
        is_loading = request.is_loading if request.is_loading is not None else False
        
        # 创建新消息
        execute_update(
            """INSERT INTO chat_messages 
               (id, session_id, message_type, content, parent_id, paired_ai_id, 
                references, question, is_loading, created_at) 
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())""",
            (
                message_id, 
                request.session_id, 
                request.message_type, 
                request.content, 
                parent_id, 
                paired_ai_id,
                references,
                question,
                is_loading
            )
        )
        
        # 更新会话的更新时间
        execute_update(
            "UPDATE chat_sessions SET updated_at = NOW() WHERE id = %s", 
            (request.session_id,)
        )
        
        return {"id": message_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 更新消息
@router.put("/chat/messages/{message_id}")
async def update_chat_message(message_id: str, request: ChatMessageCreate):
    try:
        # 检查消息是否存在
        message = execute_query("SELECT * FROM chat_messages WHERE id = %s", (message_id,))
        
        if not message:
            raise HTTPException(status_code=404, detail="消息不存在")
        
        # 处理可能为None的字段
        parent_id = request.parent_id if request.parent_id is not None else None
        paired_ai_id = request.paired_ai_id if request.paired_ai_id is not None else None
        references = request.references if request.references is not None else []
        question = request.question if request.question is not None else ""
        is_loading = request.is_loading if request.is_loading is not None else False
        
        # 更新消息
        execute_update(
            """UPDATE chat_messages 
               SET content = %s, parent_id = %s, paired_ai_id = %s, 
                   references = %s, question = %s, is_loading = %s 
               WHERE id = %s""",
            (
                request.content, 
                parent_id, 
                paired_ai_id,
                references,
                question,
                is_loading,
                message_id
            )
        )
        
        return {"code": 200, "message": "消息已更新"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 删除消息
@router.delete("/chat/messages/{message_id}")
async def delete_chat_message(message_id: str):
    try:
        # 检查消息是否存在
        message = execute_query("SELECT * FROM chat_messages WHERE id = %s", (message_id,))
        
        if not message:
            raise HTTPException(status_code=404, detail="消息不存在")
        
        # 删除消息
        execute_update("DELETE FROM chat_messages WHERE id = %s", (message_id,))
        
        return {"code": 200, "message": "消息已删除"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 