from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime
import uuid
from pydantic import BaseModel
from database import execute_query, execute_update
import json

router = APIRouter()

class ChatSessionBase(BaseModel):
    user_id: str
    title: str

class ChatSessionCreate(ChatSessionBase):
    pass

class ChatSession(ChatSessionBase):
    id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class ChatMessageBase(BaseModel):
    session_id: str
    message_type: str
    content: str
    parent_id: Optional[str] = None
    paired_ai_id: Optional[str] = None
    references: Optional[dict] = None
    question: Optional[str] = None
    is_loading: bool = False

class ChatMessageCreate(ChatMessageBase):
    pass

class ChatMessage(ChatMessageBase):
    id: str
    created_at: datetime

    class Config:
        from_attributes = True

def get_chat_sessions_db(user_id: str):
    query = """
    SELECT * FROM chat_sessions 
    WHERE user_id = %s 
    ORDER BY updated_at DESC
    """
    return execute_query(query, (user_id,))

def create_chat_session_db(session_data: dict):
    session_id = str(uuid.uuid4())
    now = datetime.now()
    
    query = """
    INSERT INTO chat_sessions 
    (id, user_id, title, created_at, updated_at) 
    VALUES (%s, %s, %s, %s, %s)
    """
    params = (
        session_id,
        session_data['user_id'],
        session_data['title'],
        now,
        now
    )
    execute_update(query, params)
    return session_id

def delete_chat_session_db(session_id: str):
    # 先删除相关的消息
    query = "DELETE FROM chat_messages WHERE session_id = %s"
    execute_update(query, (session_id,))
    
    # 然后删除会话
    query = "DELETE FROM chat_sessions WHERE id = %s"
    execute_update(query, (session_id,))

def clear_user_sessions_db(user_id: str):
    # 先获取用户的所有会话ID
    query = "SELECT id FROM chat_sessions WHERE user_id = %s"
    sessions = execute_query(query, (user_id,))
    
    # 删除所有相关消息
    for session in sessions:
        query = "DELETE FROM chat_messages WHERE session_id = %s"
        execute_update(query, (session['id'],))
    
    # 删除所有会话
    query = "DELETE FROM chat_sessions WHERE user_id = %s"
    execute_update(query, (user_id,))

def get_chat_messages_db(session_id: str):
    query = """
    SELECT * FROM chat_messages 
    WHERE session_id = %s 
    ORDER BY created_at ASC
    """
    return execute_query(query, (session_id,))

def create_chat_message_db(message_data: dict):
    message_id = str(uuid.uuid4())
    now = datetime.now()
    
    query = """
    INSERT INTO chat_messages 
    (id, session_id, message_type, content, parent_id, paired_ai_id, references, question, is_loading, created_at) 
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    params = (
        message_id,
        message_data['session_id'],
        message_data['message_type'],
        message_data['content'],
        message_data.get('parent_id'),
        message_data.get('paired_ai_id'),
        json.dumps(message_data.get('references')) if message_data.get('references') else None,
        message_data.get('question'),
        message_data.get('is_loading', False),
        now
    )
    execute_update(query, params)
    return message_id

@router.get("/sessions", response_model=List[ChatSession])
async def get_chat_sessions(
    user_id: str = Query(..., description="用户ID")
):
    sessions = get_chat_sessions_db(user_id)
    return [ChatSession(**session) for session in sessions]

@router.post("/sessions", response_model=ChatSession)
async def create_chat_session(
    session: ChatSessionCreate
):
    session_id = create_chat_session_db(session.dict())
    new_session = get_chat_sessions_db(session.user_id)[0]
    return ChatSession(**new_session)

@router.delete("/sessions/{session_id}")
async def delete_chat_session(
    session_id: str
):
    delete_chat_session_db(session_id)
    return {"message": "会话删除成功"}

@router.delete("/sessions/clear")
async def clear_user_sessions(
    user_id: str = Query(..., description="用户ID")
):
    clear_user_sessions_db(user_id)
    return {"message": "所有会话已清除"}

@router.get("/messages", response_model=List[ChatMessage])
async def get_chat_messages(
    session_id: str = Query(..., description="会话ID")
):
    messages = get_chat_messages_db(session_id)
    return [ChatMessage(**message) for message in messages]

@router.post("/messages", response_model=ChatMessage)
async def create_chat_message(
    message: ChatMessageCreate
):
    message_id = create_chat_message_db(message.dict())
    new_message = get_chat_messages_db(message.session_id)[-1]
    return ChatMessage(**new_message) 