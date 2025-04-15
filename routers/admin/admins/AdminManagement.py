#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
管理员管理模块
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Path, Body
from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field, validator
import re
import hashlib
from datetime import datetime
from database import execute_query, execute_update
from routers.user.login import get_current_admin
import logging
import json

# 初始化 APIRouter 实例
router = APIRouter()

# 初始化日志记录器
logger = logging.getLogger(__name__)

# 请求模型
class AdminCreate(BaseModel):
    phone_number: str = Field(..., description="手机号")
    full_name: str = Field(..., description="姓名")
    email: EmailStr = Field(..., description="邮箱")
    role: str = Field(..., description="角色: admin-管理员, operator-操作员")
    status: int = Field(1, description="状态: 0-禁用, 1-正常")
    password: str = Field(..., description="密码")
    
    @validator('phone_number')
    def validate_phone_number(cls, v):
        if not re.match(r'^1[3-9]\d{9}$', v):
            raise ValueError('请输入有效的手机号')
        return v
    
    @validator('role')
    def validate_role(cls, v):
        if v not in ['admin', 'operator']:
            raise ValueError('角色必须是 admin 或 operator')
        return v
    
    @validator('status')
    def validate_status(cls, v):
        if v not in [0, 1]:
            raise ValueError('状态必须是 0 或 1')
        return v
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 6:
            raise ValueError('密码长度至少为6个字符')
        return v


class AdminUpdate(BaseModel):
    phone_number: str = Field(..., description="手机号")
    full_name: str = Field(..., description="姓名")
    email: EmailStr = Field(..., description="邮箱")
    role: str = Field(..., description="角色: admin-管理员, operator-操作员")
    status: int = Field(..., description="状态: 0-禁用, 1-正常")
    
    @validator('phone_number')
    def validate_phone_number(cls, v):
        if not re.match(r'^1[3-9]\d{9}$', v):
            raise ValueError('请输入有效的手机号')
        return v
    
    @validator('role')
    def validate_role(cls, v):
        if v not in ['admin', 'operator']:
            raise ValueError('角色必须是 admin 或 operator')
        return v
    
    @validator('status')
    def validate_status(cls, v):
        if v not in [0, 1]:
            raise ValueError('状态必须是 0 或 1')
        return v


class AdminStatusUpdate(BaseModel):
    admin_id: int = Field(..., description="管理员ID")
    status: int = Field(..., description="状态: 0-禁用, 1-正常")
    admin_id_operator: Optional[int] = None  # 操作者管理员ID


# 辅助函数
def md5_password(password: str) -> str:
    """
    对密码进行MD5加密
    """
    return hashlib.md5(password.encode('utf-8')).hexdigest()


# API路由
@router.get("/admin/admins", tags=["管理员管理"])
async def get_admins(
    page: int = Query(1, description="页码"),
    page_size: int = Query(10, description="每页数量"),
    phone_number: Optional[str] = Query(None, description="手机号筛选"),
    full_name: Optional[str] = Query(None, description="姓名筛选"),
    role: Optional[str] = Query(None, description="角色筛选"),
    status: Optional[str] = Query(None, description="状态筛选"),
    admin_id: Optional[int] = Query(None, description="操作管理员ID")
):
    """
    获取管理员列表，支持分页和条件筛选
    """
    try:
        # 计算偏移量
        offset = (page - 1) * page_size
        
        # 构建基础查询SQL
        query = """
            SELECT 
                admin_id, phone_number, full_name, email, role, status, 
                last_login_time, created_at, updated_at
            FROM 
                admins
            WHERE 1=1
        """
        params = []
        
        # 添加筛选条件
        if phone_number:
            query += " AND phone_number LIKE %s"
            params.append(f"%{phone_number}%")
        
        if full_name:
            query += " AND full_name LIKE %s"
            params.append(f"%{full_name}%")
        
        if role:
            query += " AND role = %s"
            params.append(role)
            
        if status:
            query += " AND status = %s"
            params.append(int(status))
        
        # 查询符合条件的总记录数
        count_query = f"SELECT COUNT(*) as count FROM ({query}) as filtered_admins"
        count_result = execute_query(count_query, tuple(params))
        total_count = count_result[0]['count'] if count_result else 0
        
        # 添加排序和分页
        query += " ORDER BY admin_id DESC LIMIT %s OFFSET %s"
        params.append(page_size)
        params.append(offset)
        
        # 查询管理员列表
        admin_list = execute_query(query, tuple(params))
        
        # 处理日期时间格式
        for admin in admin_list:
            admin['last_login_time'] = admin['last_login_time'].strftime("%Y-%m-%d %H:%M:%S") if admin['last_login_time'] else ""
            admin['created_at'] = admin['created_at'].strftime("%Y-%m-%d %H:%M:%S") if admin['created_at'] else ""
            admin['updated_at'] = admin['updated_at'].strftime("%Y-%m-%d %H:%M:%S") if admin['updated_at'] else ""
        
        # 记录操作日志
        try:
            log_query = """
                INSERT INTO admin_logs (log_type, operation, admin_id, details, created_at) 
                VALUES (%s, %s, %s, %s, %s)
            """
            execute_update(log_query, (
                'ADMIN_MANAGEMENT', '查询管理员列表', admin_id, f'查询了管理员列表，过滤条件: {status}', datetime.now()
            ))
        except Exception as log_error:
            # 记录错误但不影响主要流程
            # print(f"记录操作日志失败: {str(log_error)}")
            logger.error(f"记录操作日志失败: {str(log_error)}")
        
        return {
            "code": 200,
            "message": "获取管理员列表成功",
            "data": {
                "admins": admin_list,
                "total": total_count,
                "page": page,
                "page_size": page_size
            }
        }
    except Exception as e:
        # 记录错误日志
        # print(f"获取管理员列表失败: {str(e)}")
        logger.error(f"获取管理员列表失败: {str(e)}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"获取管理员列表失败: {str(e)}")


@router.post("/admin/admins", tags=["管理员管理"])
async def create_admin(admin_data: AdminCreate, admin_id: Optional[int] = Query(None, description="操作管理员ID")):
    """
    添加新管理员
    """
    try:
        # 检查手机号是否已存在
        check_query = "SELECT admin_id FROM admins WHERE phone_number = %s"
        existing_admin = execute_query(check_query, (admin_data.phone_number,))
        
        if existing_admin:
            return {
                "code": 400,
                "message": "手机号已存在"
            }
        
        # 创建新管理员
        insert_query = """
            INSERT INTO admins (phone_number, password, full_name, email, role, status)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        
        # 使用MD5加密密码
        encrypted_password = md5_password(admin_data.password)
        
        new_admin_id = execute_update(
            insert_query, 
            (
                admin_data.phone_number,
                encrypted_password,
                admin_data.full_name,
                admin_data.email,
                admin_data.role,
                admin_data.status
            )
        )
        
        # 记录操作日志
        if admin_id:
            try:
                execute_update(
                    """INSERT INTO operation_logs (admin_id, operation_type, operation_desc, created_at) 
                       VALUES (%s, %s, %s, NOW())""", 
                    (admin_id, "创建", f"管理员{admin_id}创建了管理员: {admin_data.full_name}({admin_data.phone_number})")
                )
            except Exception as log_error:
                # 仅记录日志错误，不影响主流程
                # print(f"记录操作日志失败: {str(log_error)}")
                logger.error(f"记录操作日志失败: {str(log_error)}")
        
        return {
            "code": 200,
            "message": "添加管理员成功",
            "data": {
                "admin_id": new_admin_id
            }
        }
    except Exception as e:
        # 记录错误日志
        # print(f"添加管理员失败: {str(e)}")
        logger.error(f"添加管理员失败: {str(e)}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"添加管理员失败: {str(e)}")


@router.put("/admin/admins/{admin_id}", tags=["管理员管理"])
async def update_admin(
    admin_id: int = Path(..., description="管理员ID"),
    admin_data: AdminUpdate = Body(...),
    current_admin_id: Optional[int] = Query(None, description="操作管理员ID")
):
    """
    更新管理员信息
    """
    try:
        # 检查管理员是否存在
        check_query = "SELECT * FROM admins WHERE admin_id = %s"
        admin = execute_query(check_query, (admin_id,))
        
        if not admin:
            return {
                "code": 404,
                "message": "管理员不存在"
            }
        
        # 检查手机号是否已被其他管理员使用
        if admin[0]['phone_number'] != admin_data.phone_number:
            check_phone_query = "SELECT * FROM admins WHERE phone_number = %s AND admin_id != %s"
            existing_admin = execute_query(check_phone_query, (admin_data.phone_number, admin_id))
            
            if existing_admin:
                return {
                    "code": 400,
                    "message": "手机号已被其他管理员使用"
                }
        
        # 更新管理员信息
        update_query = """
            UPDATE admins 
            SET phone_number = %s, full_name = %s, email = %s, role = %s, status = %s 
            WHERE admin_id = %s
        """
        
        execute_update(
            update_query, 
            (
                admin_data.phone_number,
                admin_data.full_name,
                admin_data.email,
                admin_data.role,
                admin_data.status,
                admin_id
            )
        )
        
        # 记录操作日志
        if current_admin_id:
            try:
                execute_update(
                    """INSERT INTO operation_logs (admin_id, operation_type, operation_desc, created_at) 
                       VALUES (%s, %s, %s, NOW())""", 
                    (current_admin_id, "更新", f"管理员{current_admin_id}更新了管理员信息: {admin_data.full_name}({admin_data.phone_number})")
                )
            except Exception as log_error:
                # 仅记录日志错误，不影响主流程
                # print(f"记录操作日志失败: {str(log_error)}")
                logger.error(f"记录操作日志失败: {str(log_error)}")
        
        return {
            "code": 200,
            "message": "更新管理员信息成功"
        }
    except Exception as e:
        # 记录错误日志
        # print(f"更新管理员信息失败: {str(e)}")
        logger.error(f"更新管理员信息失败: {str(e)}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"更新管理员信息失败: {str(e)}")


@router.post("/admin/admins/status", tags=["管理员管理"])
async def update_admin_status(status_data: AdminStatusUpdate):
    """
    修改管理员状态（启用/禁用）
    """
    try:
        # 检查管理员是否存在
        check_query = "SELECT * FROM admins WHERE admin_id = %s"
        admin = execute_query(check_query, (status_data.admin_id,))
        
        if not admin:
            return {
                "code": 404,
                "message": "管理员不存在"
            }
        
        # 不能修改自己的状态
        if status_data.admin_id_operator and status_data.admin_id_operator == status_data.admin_id:
            return {
                "code": 400,
                "message": "不能修改自己的状态"
            }
        
        # 更新状态
        update_query = "UPDATE admins SET status = %s WHERE admin_id = %s"
        execute_update(update_query, (status_data.status, status_data.admin_id))
        
        status_text = "启用" if status_data.status == 1 else "禁用"
        
        # 记录操作日志
        if status_data.admin_id_operator:
            try:
                execute_update(
                    """INSERT INTO operation_logs (admin_id, operation_type, operation_desc, created_at) 
                       VALUES (%s, %s, %s, NOW())""", 
                    (
                        status_data.admin_id_operator, 
                        f"更新", 
                        f"管理员{status_data.admin_id_operator}{status_text}了管理员账号: {admin[0]['full_name']}({admin[0]['phone_number']})"
                    )
                )
            except Exception as log_error:
                # 仅记录日志错误，不影响主流程
                # print(f"记录操作日志失败: {str(log_error)}")
                logger.error(f"记录操作日志失败: {str(log_error)}")
        
        return {
            "code": 200,
            "message": f"管理员账号{status_text}成功"
        }
    except Exception as e:
        # 记录错误日志
        # print(f"修改管理员状态失败: {str(e)}")
        logger.error(f"修改管理员状态失败: {str(e)}")
        # 返回错误响应
        raise HTTPException(status_code=500, detail=f"修改管理员状态失败: {str(e)}")
