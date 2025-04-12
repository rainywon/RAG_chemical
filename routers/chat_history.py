# 引入 FastAPI 中的 APIRouter 和 HTTPException 模块，用于创建路由和处理异常
from fastapi import APIRouter, HTTPException, Depends
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
# 引入用户认证依赖函数
from routers.login import get_current_user

# 初始化 APIRouter 实例，用于定义路由
router = APIRouter()

# 定义请求和响应模型
class ChatSessionCreate(BaseModel):
    """创建聊天会话的请求模型"""
    user_id: str
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

# API 路由

@router.post("/chat/sessions", tags=["聊天历史"])
async def create_chat_session(session: ChatSessionCreate):
    """
    创建新的聊天会话。
    """
    try:
        # 生成唯一的会话ID
        session_id = str(uuid.uuid4())
        
        # 如果没有提供标题，则使用默认标题
        title = session.title or f"对话 {datetime.now().strftime('%H:%M:%S')}"
        
        # 在数据库中创建会话
        execute_update(
            """INSERT INTO chat_sessions (id, user_id, title, created_at) 
               VALUES (%s, %s, %s, NOW())""", 
            (session_id, session.user_id, title)
        )
        
        return {"id": session_id, "title": title}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建会话失败: {str(e)}")

@router.get("/chat/sessions/{user_id}", tags=["聊天历史"])
async def get_user_chat_sessions(user_id: str):
    """
    获取用户的所有聊天会话。
    """
    try:
        # 查询用户的所有会话
        sessions = execute_query(
            """SELECT id, title, created_at, updated_at 
               FROM chat_sessions 
               WHERE user_id = %s 
               ORDER BY updated_at DESC""", 
            (user_id,)
        )
        
        # 转换datetime对象为字符串，以便JSON序列化
        for session in sessions:
            session['created_at'] = session['created_at'].isoformat() if session['created_at'] else None
            session['updated_at'] = session['updated_at'].isoformat() if session['updated_at'] else None
        
        return {"sessions": sessions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取会话列表失败: {str(e)}")

@router.get("/chat/sessions/{session_id}/messages", tags=["聊天历史"])
async def get_chat_session_messages(session_id: str):
    """
    获取特定聊天会话的所有消息。
    """
    try:
        # 查询会话的所有消息
        messages = execute_query(
            """SELECT id, message_type, content, parent_id, paired_ai_id, 
                      message_references, question, is_loading, created_at
               FROM chat_messages 
               WHERE session_id = %s 
               ORDER BY created_at ASC""", 
            (session_id,)
        )
        
        # 转换datetime对象为字符串，以便JSON序列化
        for message in messages:
            message['created_at'] = message['created_at'].isoformat() if message['created_at'] else None
            # 确保布尔值正确转换
            message['is_loading'] = bool(message['is_loading'])
            
            # 处理 message_references 字段
            # 如果已经是字符串，尝试解析为 JSON 对象
            if isinstance(message['message_references'], str):
                try:
                    message['message_references'] = json.loads(message['message_references'])
                except json.JSONDecodeError:
                    # 如果无法解析为 JSON，保持原样
                    pass
        
        return {"messages": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取消息列表失败: {str(e)}")

@router.post("/chat/messages", tags=["聊天历史"])
async def create_chat_message(message: ChatMessageCreate):
    """
    创建新的聊天消息。
    """
    try:

        
        # 确保 message_references 是有效的 JSON 字符串
        if message.message_references is None:
            message_references = '{}'
        else:
            # 我们已经在模型中定义了它是字符串类型，所以直接使用
            message_references = message.message_references
            
            # 验证它是有效的 JSON 字符串
            try:
                json.loads(message_references)
            except json.JSONDecodeError:
                # 如果不是有效的 JSON，使用空对象
                message_references = '{}'
        
        # 在数据库中创建消息
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
                message_references,  # 已经是 JSON 字符串
                message.question,
                message.is_loading
            )
        )
        
        # 更新会话的更新时间
        execute_update(
            """UPDATE chat_sessions SET updated_at = NOW() WHERE id = %s""", 
            (message.session_id,)
        )
        
        return {"id": message.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建消息失败: {str(e)}")

@router.put("/chat/messages/{message_id}", tags=["聊天历史"])
async def update_chat_message(message_id: str, message: ChatMessageUpdate):
    """
    更新现有的聊天消息。
    """
    try:
        # 构建更新SQL语句
        update_fields = []
        params = []
        
        if message.content is not None:
            update_fields.append("content = %s")
            params.append(message.content)
            
        if message.message_references is not None:
            update_fields.append("message_references = %s")
            
            # 验证 message_references 是有效的 JSON 字符串
            try:
                json.loads(message.message_references)
                message_references = message.message_references
            except json.JSONDecodeError:
                # 如果不是有效的 JSON，使用空对象
                message_references = '{}'
                
            params.append(message_references)
            
        if message.is_loading is not None:
            update_fields.append("is_loading = %s")
            params.append(message.is_loading)
        
        if not update_fields:
            return {"message": "Nothing to update"}
        
        # 执行更新操作
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
async def update_chat_session(session_id: str, session: ChatSessionUpdate):
    """
    更新聊天会话信息，例如标题。
    """
    try:
        if session.title:
            execute_update(
                """UPDATE chat_sessions SET title = %s, updated_at = NOW() WHERE id = %s""", 
                (session.title, session_id)
            )
            
        return {"message": "Session updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新会话失败: {str(e)}")

@router.delete("/chat/sessions/{session_id}", tags=["聊天历史"])
async def delete_chat_session(session_id: str):
    """
    删除聊天会话及其所有消息。
    """
    try:
        # 删除会话及其消息（依赖外键约束自动删除消息）
        result = execute_update(
            """DELETE FROM chat_sessions WHERE id = %s""", 
            (session_id,)
        )
        
        if result == 0:
            raise HTTPException(status_code=404, detail="Chat session not found")
            
        return {"message": "Session deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除会话失败: {str(e)}")

@router.delete("/chat/sessions/user/{user_id}", tags=["聊天历史"])
async def delete_all_user_chat_sessions(user_id: str):
    """
    删除用户的所有聊天会话及其消息。
    """
    try:
        # 删除用户的所有会话（依赖外键约束自动删除消息）
        result = execute_update(
            """DELETE FROM chat_sessions WHERE user_id = %s""", 
            (user_id,)
        )
            
        return {"message": f"Deleted {result} sessions successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除会话失败: {str(e)}")
