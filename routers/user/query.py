from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import asyncio
from fastapi.responses import StreamingResponse
from fastapi.concurrency import run_in_threadpool
from rag_system import RAGSystem
from config import Config
from typing import AsyncGenerator, Generator
import anyio

router = APIRouter()
config = Config()
rag_system = RAGSystem(config)


class QueryRequest(BaseModel):
    question: str


def sync_rag_generator(question: str) -> Generator[str, None, None]:
    """同步生成器包装"""
    yield from rag_system.stream_query_rag_with_kb(question)


@router.post("/query_rag/")
async def query(request: QueryRequest):
    try:
        question = request.question.strip()

        async def async_generator() -> AsyncGenerator[str, None]:
            gen = sync_rag_generator(question)
            try:
                while True:
                    try:
                        # 使用run_in_threadpool处理每个chunk
                        chunk = await run_in_threadpool(next, gen)
                        yield chunk
                    except StopIteration:
                        break
                    except RuntimeError as e:
                        if "StopIteration" in str(e):
                            break
                        raise
            except anyio.get_cancelled_exc_class() as e:
                # 处理客户端提前断开连接
                gen.close()
                raise
            except Exception as e:
                yield f"\n⚠️ 生成错误: {str(e)}"
            finally:
                # 确保生成器正确关闭
                gen.close()

        return StreamingResponse(async_generator(), media_type="text/plain")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理请求时出错: {str(e)}")


@router.post("/query_model/")
async def query_model(request: QueryRequest):
    try:
        question = request.question.strip()

        def sync_model_generator():
            yield from rag_system.stream_query_model(question)

        async def model_async_generator() -> AsyncGenerator[str, None]:
            gen = sync_model_generator()
            try:
                while True:
                    try:
                        chunk = await run_in_threadpool(next, gen)
                        yield chunk
                    except StopIteration:
                        break
                    except RuntimeError as e:
                        if "StopIteration" in str(e):
                            break
                        raise
            except anyio.get_cancelled_exc_class():
                gen.close()
                raise
            finally:
                gen.close()

        return StreamingResponse(model_async_generator(), media_type="text/plain")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理请求时出错: {str(e)}")


class ChatQueryRequest(BaseModel):
    session_id: str
    question: str


@router.post("/query_rag_chat/")
async def query_rag_chat(request: ChatQueryRequest):
    """处理带知识库的多轮对话查询"""
    try:
        question = request.question.strip()
        session_id = request.session_id

        # 从数据库获取聊天历史
        from database import execute_query
        
        # 获取会话的历史消息（按时间顺序）
        chat_history = execute_query(
            """SELECT id, message_type, content, parent_id, paired_ai_id, created_at
               FROM chat_messages 
               WHERE session_id = %s 
               ORDER BY created_at ASC""", 
            (session_id,)
        )

        def sync_rag_chat_generator():
            yield from rag_system.stream_query_with_history(session_id, question, chat_history)

        async def rag_chat_async_generator() -> AsyncGenerator[str, None]:
            gen = sync_rag_chat_generator()
            try:
                while True:
                    try:
                        chunk = await run_in_threadpool(next, gen)
                        yield chunk
                    except StopIteration:
                        break
                    except RuntimeError as e:
                        if "StopIteration" in str(e):
                            break
                        raise
            except anyio.get_cancelled_exc_class():
                gen.close()
                raise
            except Exception as e:
                yield f"\n⚠️ 生成错误: {str(e)}"
            finally:
                gen.close()

        return StreamingResponse(rag_chat_async_generator(), media_type="text/plain")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理RAG多轮对话请求时出错: {str(e)}")


@router.post("/query_model_chat/")
async def query_model_chat(request: ChatQueryRequest):
    """处理直接生成的多轮对话查询（不使用知识库）"""
    try:
        question = request.question.strip()
        session_id = request.session_id

        # 从数据库获取聊天历史
        from database import execute_query
        
        # 获取会话的历史消息（按时间顺序）
        chat_history = execute_query(
            """SELECT id, message_type, content, parent_id, paired_ai_id, created_at
               FROM chat_messages 
               WHERE session_id = %s 
               ORDER BY created_at ASC""", 
            (session_id,)
        )

        def sync_model_chat_generator():
            yield from rag_system.stream_query_model_with_history(session_id, question, chat_history)

        async def model_chat_async_generator() -> AsyncGenerator[str, None]:
            gen = sync_model_chat_generator()
            try:
                while True:
                    try:
                        chunk = await run_in_threadpool(next, gen)
                        yield chunk
                    except StopIteration:
                        break
                    except RuntimeError as e:
                        if "StopIteration" in str(e):
                            break
                        raise
            except anyio.get_cancelled_exc_class():
                gen.close()
                raise
            except Exception as e:
                yield f"\n⚠️ 生成错误: {str(e)}"
            finally:
                gen.close()

        return StreamingResponse(model_chat_async_generator(), media_type="text/plain")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理直接多轮对话请求时出错: {str(e)}")