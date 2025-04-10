from fastapi import APIRouter, HTTPException, Depends, Body, Request
from fastapi.responses import JSONResponse
import uuid
import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
from database import execute_update, execute_query
from routers.login import get_current_user

# 初始化路由
router = APIRouter()

# 定义请求/响应模型
class ChatCreate(BaseModel):
    title: str = "新的对话"

class ChatDelete(BaseModel):
    chat_id: str

class MessageAdd(BaseModel):
    chat_id: str
    message_type: str  # 'user' 或 'assistant'
    content: str

class TitleUpdate(BaseModel):
    chat_id: str
    title: str

# 获取用户历史记录列表
@router.get("/api/chat/history")
async def get_chat_history(current_user: int = Depends(get_current_user)):
    try:
        history = execute_query(
            """SELECT chat_id, title, create_time, update_time FROM chat_history 
               WHERE user_id = %s AND is_deleted = FALSE ORDER BY update_time DESC""",
            (current_user,)
        )
        
        # 格式化日期时间
        for item in history:
            item['create_time'] = item['create_time'].strftime('%Y-%m-%d %H:%M:%S')
            item['update_time'] = item['update_time'].strftime('%Y-%m-%d %H:%M:%S')
        
        return {"code": 200, "data": history}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取历史记录失败：{str(e)}")

# 创建新的聊天记录
@router.post("/api/chat/create")
async def create_chat(chat_data: ChatCreate, current_user: int = Depends(get_current_user)):
    try:
        chat_id = str(uuid.uuid4())
        now = datetime.datetime.now()
        
        execute_update(
            """INSERT INTO chat_history (user_id, chat_id, title, create_time, update_time) 
               VALUES (%s, %s, %s, %s, %s)""",
            (current_user, chat_id, chat_data.title, now, now)
        )
        
        return {
            "code": 200, 
            "data": {
                "chat_id": chat_id,
                "title": chat_data.title,
                "create_time": now.strftime('%Y-%m-%d %H:%M:%S'),
                "update_time": now.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建聊天失败：{str(e)}")

# 删除聊天记录
@router.post("/api/chat/delete")
async def delete_chat(chat_data: ChatDelete, current_user: int = Depends(get_current_user)):
    try:
        # 检查该聊天是否属于当前用户
        result = execute_query(
            """SELECT COUNT(*) as count FROM chat_history WHERE user_id = %s AND chat_id = %s""",
            (current_user, chat_data.chat_id)
        )
        
        if not result or result[0]['count'] == 0:
            raise HTTPException(status_code=403, detail="无权操作此聊天记录")
        
        # 逻辑删除（将is_deleted标记为TRUE）
        execute_update(
            """UPDATE chat_history SET is_deleted = TRUE WHERE chat_id = %s""",
            (chat_data.chat_id,)
        )
        
        return {"code": 200, "message": "删除成功"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除聊天失败：{str(e)}")

# 获取聊天消息
@router.get("/api/chat/messages")
async def get_chat_messages(chat_id: str, current_user: int = Depends(get_current_user)):
    try:
        if not chat_id:
            raise HTTPException(status_code=400, detail="缺少chat_id参数")
        
        # 验证用户是否有权限访问该聊天
        result = execute_query(
            """SELECT COUNT(*) as count FROM chat_history WHERE user_id = %s AND chat_id = %s""",
            (current_user, chat_id)
        )
        
        if not result or result[0]['count'] == 0:
            raise HTTPException(status_code=403, detail="无权访问此聊天记录")
        
        # 获取聊天消息
        messages = execute_query(
            """SELECT id, message_type, content, send_time FROM chat_messages 
               WHERE chat_id = %s ORDER BY send_time ASC""",
            (chat_id,)
        )
        
        # 格式化日期时间
        for msg in messages:
            msg['send_time'] = msg['send_time'].strftime('%Y-%m-%d %H:%M:%S')
        
        return {"code": 200, "data": messages}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取聊天消息失败：{str(e)}")

# 添加聊天消息
@router.post("/api/chat/message/add")
async def add_chat_message(message_data: MessageAdd, current_user: int = Depends(get_current_user)):
    try:
        # 验证用户是否有权限访问该聊天
        result = execute_query(
            """SELECT COUNT(*) as count FROM chat_history WHERE user_id = %s AND chat_id = %s""",
            (current_user, message_data.chat_id)
        )
        
        if not result or result[0]['count'] == 0:
            raise HTTPException(status_code=403, detail="无权操作此聊天记录")
        
        # 添加消息
        now = datetime.datetime.now()
        execute_update(
            """INSERT INTO chat_messages (chat_id, message_type, content, send_time) 
               VALUES (%s, %s, %s, %s)""",
            (message_data.chat_id, message_data.message_type, message_data.content, now)
        )
        
        # 获取最后插入的ID
        last_id = execute_query("SELECT LAST_INSERT_ID() as id")[0]['id']
        
        # 更新聊天记录的更新时间
        execute_update(
            """UPDATE chat_history SET update_time = %s WHERE chat_id = %s""",
            (now, message_data.chat_id)
        )
        
        return {
            "code": 200, 
            "data": {
                "id": last_id,
                "chat_id": message_data.chat_id,
                "message_type": message_data.message_type,
                "content": message_data.content,
                "send_time": now.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"添加消息失败：{str(e)}")

# 更新聊天标题
@router.post("/api/chat/update_title")
async def update_chat_title(title_data: TitleUpdate, current_user: int = Depends(get_current_user)):
    try:
        # 验证用户是否有权限操作该聊天
        result = execute_query(
            """SELECT COUNT(*) as count FROM chat_history WHERE user_id = %s AND chat_id = %s""",
            (current_user, title_data.chat_id)
        )
        
        if not result or result[0]['count'] == 0:
            raise HTTPException(status_code=403, detail="无权操作此聊天记录")
        
        # 更新标题
        execute_update(
            """UPDATE chat_history SET title = %s, update_time = %s WHERE chat_id = %s""",
            (title_data.title, datetime.datetime.now(), title_data.chat_id)
        )
        
        return {"code": 200, "message": "更新标题成功"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新标题失败：{str(e)}")
