#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
管理员个人资料和密码管理模块
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body, Request
from typing import Optional
from pydantic import BaseModel, EmailStr, Field, validator
import re
import hashlib
from datetime import datetime
import traceback
from database import execute_query, execute_update

# 初始化 APIRouter 实例
router = APIRouter()

# 请求模型
class AdminProfileUpdate(BaseModel):
    admin_id: Optional[int] = Field(None, description="管理员ID (可选，将从token中获取)")
    full_name: str = Field(..., description="姓名")
    email: EmailStr = Field(..., description="邮箱")
    phone_number: str = Field(..., description="手机号")
    
    @validator('full_name')
    def validate_full_name(cls, v):
        if not v.strip():
            raise ValueError('姓名不能为空')
        return v
    
    @validator('phone_number')
    def validate_phone_number(cls, v):
        if not re.match(r'^1[3-9]\d{9}$', v):
            raise ValueError('请输入正确的手机号格式')
        return v

class AdminPasswordUpdate(BaseModel):
    admin_id: Optional[int] = Field(None, description="管理员ID (可选，将从token中获取)")
    old_password: str = Field(..., description="旧密码")
    new_password: str = Field(..., description="新密码")
    confirm_password: str = Field(..., description="确认新密码")
    
    @validator('new_password')
    def validate_new_password(cls, v):
        if len(v) < 6:
            raise ValueError('新密码长度至少为6个字符')
        return v
    
    @validator('confirm_password')
    def validate_confirm_password(cls, v, values):
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('两次输入的密码不一致')
        return v


# 辅助函数
def md5_password(password: str) -> str:
    """
    对密码进行MD5加密
    """
    return hashlib.md5(password.encode('utf-8')).hexdigest()


# API路由
@router.get("/admin/profile", tags=["管理员个人信息"])
async def get_admin_profile(request: Request):
    """
    获取管理员的个人资料（通过Token认证）
    """
    try:
        # 从Authorization头获取token
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return {
                "code": 401,
                "message": "无效的认证信息"
            }
        
        token = auth_header.split(' ')[1]
        
        # 查询token对应的管理员ID
        try:
            query = """
                SELECT admin_id 
                FROM admin_tokens 
                WHERE token = %s AND is_valid = 1 AND expire_at > NOW()
            """
            token_result = execute_query(query, (token,))
            
            if not token_result:
                return {
                    "code": 401,
                    "message": "无效的token或token已过期"
                }
            
            admin_id = token_result[0]['admin_id']
            
            # 查询管理员信息
            query = """
                SELECT 
                    admin_id, phone_number, full_name, email, role, status, 
                    last_login_time, created_at, updated_at
                FROM 
                    admins
                WHERE 
                    admin_id = %s
            """
            admin_info = execute_query(query, (admin_id,))
            
            if not admin_info:
                return {
                    "code": 404,
                    "message": "管理员信息不存在"
                }
            
            admin_data = admin_info[0]
            
            # 处理日期时间格式
            admin_data['last_login_time'] = admin_data['last_login_time'].strftime("%Y-%m-%d %H:%M:%S") if admin_data['last_login_time'] else ""
            admin_data['created_at'] = admin_data['created_at'].strftime("%Y-%m-%d %H:%M:%S") if admin_data['created_at'] else ""
            admin_data['updated_at'] = admin_data['updated_at'].strftime("%Y-%m-%d %H:%M:%S") if admin_data['updated_at'] else ""
            
        except Exception as e:
            # 记录错误日志
            traceback_str = traceback.format_exc()
            print(f"查询管理员信息失败: {str(e)}")
            print(f"错误详情: {traceback_str}")
            return {
                "code": 500,
                "message": f"查询管理员信息失败: {str(e)}"
            }
        
        # 记录操作日志
        try:
            execute_update(
                """INSERT INTO operation_logs (admin_id, operation_type, operation_desc, created_at) 
                   VALUES (%s, %s, %s, NOW())""", 
                (admin_id, "查询", f"管理员{admin_data['full_name']}查询个人资料")
            )
        except Exception as log_error:
            # 仅记录日志错误，不影响主流程
            print(f"记录操作日志失败: {str(log_error)}")
        
        return {
            "code": 200,
            "message": "获取个人资料成功",
            "data": admin_data
        }
    except Exception as e:
        # 记录错误日志
        traceback_str = traceback.format_exc()
        print(f"获取管理员个人资料失败: {str(e)}")
        print(f"错误详情: {traceback_str}")
        # 返回错误响应
        return {
            "code": 500,
            "message": f"获取管理员个人资料失败: {str(e)}"
        }


@router.put("/admin/profile", tags=["管理员个人信息"])
async def update_admin_profile(request: Request, profile_data: AdminProfileUpdate):
    """
    更新管理员的个人资料（通过Token认证）
    """
    try:
        # 从Authorization头获取token
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return {
                "code": 401,
                "message": "无效的认证信息"
            }
        
        token = auth_header.split(' ')[1]
        
        # 查询token对应的管理员ID
        try:
            query = """
                SELECT admin_id 
                FROM admin_tokens 
                WHERE token = %s AND is_valid = 1 AND expire_at > NOW()
            """
            token_result = execute_query(query, (token,))
            
            if not token_result:
                return {
                    "code": 401,
                    "message": "无效的token或token已过期"
                }
            
            admin_id = token_result[0]['admin_id']
            
            # 确保请求中的admin_id与token中的一致（如果请求中提供了admin_id）
            if profile_data.admin_id is not None and profile_data.admin_id != admin_id:
                return {
                    "code": 403,
                    "message": "无权修改其他管理员的信息"
                }
                
            # 设置admin_id（如果请求中没有提供）
            if profile_data.admin_id is None:
                profile_data.admin_id = admin_id
        except Exception as e:
            # 记录错误日志
            traceback_str = traceback.format_exc()
            print(f"查询管理员信息失败: {str(e)}")
            print(f"错误详情: {traceback_str}")
            return {
                "code": 500,
                "message": f"认证失败: {str(e)}"
            }
        
        # 先检查手机号是否已被其他管理员使用
        try:
            check_query = """
                SELECT admin_id FROM admins 
                WHERE phone_number = %s AND admin_id != %s
            """
            existing_admin = execute_query(check_query, (profile_data.phone_number, admin_id))
            
            if existing_admin:
                return {
                    "code": 400,
                    "message": "该手机号已被其他管理员使用，请更换手机号"
                }
        except Exception as e:
            # 记录错误日志
            traceback_str = traceback.format_exc()
            print(f"检查手机号唯一性失败: {str(e)}")
            print(f"错误详情: {traceback_str}")
            return {
                "code": 500,
                "message": f"检查手机号失败: {str(e)}"
            }
        
        # 获取管理员姓名用于日志记录
        try:
            query = "SELECT full_name FROM admins WHERE admin_id = %s"
            admin_info = execute_query(query, (admin_id,))
            admin_name = admin_info[0]['full_name'] if admin_info else f"管理员{admin_id}"
        except Exception as e:
            admin_name = f"管理员{admin_id}"
            print(f"获取管理员姓名失败: {str(e)}")
        
        # 更新管理员信息
        try:
            update_query = """
                UPDATE admins 
                SET full_name = %s, email = %s, phone_number = %s, updated_at = NOW()
                WHERE admin_id = %s
            """
            affected_rows = execute_update(update_query, (profile_data.full_name, profile_data.email, profile_data.phone_number, admin_id))
            
            if affected_rows == 0:
                return {
                    "code": 404,
                    "message": "管理员信息不存在，更新失败"
                }
        except Exception as e:
            # 记录错误日志
            traceback_str = traceback.format_exc()
            print(f"更新管理员信息失败: {str(e)}")
            print(f"错误详情: {traceback_str}")
            return {
                "code": 500,
                "message": f"更新管理员信息失败: {str(e)}"
            }
        
        # 记录操作日志
        try:
            execute_update(
                """INSERT INTO operation_logs (admin_id, operation_type, operation_desc, created_at) 
                   VALUES (%s, %s, %s, NOW())""", 
                (admin_id, "更新", f"{admin_name}更新了个人资料")
            )
        except Exception as log_error:
            # 仅记录日志错误，不影响主流程
            print(f"记录操作日志失败: {str(log_error)}")
        
        return {
            "code": 200,
            "message": "更新个人资料成功"
        }
    except Exception as e:
        # 记录错误日志
        traceback_str = traceback.format_exc()
        print(f"更新管理员个人资料失败: {str(e)}")
        print(f"错误详情: {traceback_str}")
        # 返回错误响应
        return {
            "code": 500,
            "message": f"更新管理员个人资料失败: {str(e)}"
        }


@router.put("/admin/password", tags=["管理员个人信息"])
async def update_admin_password(request: Request, password_data: AdminPasswordUpdate):
    """
    修改管理员的密码（通过Token认证）
    """
    try:
        # 从Authorization头获取token
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return {
                "code": 401,
                "message": "无效的认证信息"
            }
        
        token = auth_header.split(' ')[1]
        
        # 查询token对应的管理员ID
        try:
            query = """
                SELECT admin_id 
                FROM admin_tokens 
                WHERE token = %s AND is_valid = 1 AND expire_at > NOW()
            """
            token_result = execute_query(query, (token,))
            
            if not token_result:
                return {
                    "code": 401,
                    "message": "无效的token或token已过期"
                }
            
            admin_id = token_result[0]['admin_id']
            
            # 确保请求中的admin_id与token中的一致（如果请求中提供了admin_id）
            if password_data.admin_id is not None and password_data.admin_id != admin_id:
                return {
                    "code": 403,
                    "message": "无权修改其他管理员的密码"
                }
                
            # 设置admin_id（如果请求中没有提供）
            if password_data.admin_id is None:
                password_data.admin_id = admin_id
        except Exception as e:
            # 记录错误日志
            traceback_str = traceback.format_exc()
            print(f"查询管理员信息失败: {str(e)}")
            print(f"错误详情: {traceback_str}")
            return {
                "code": 500,
                "message": f"认证失败: {str(e)}"
            }
        
        # 检查旧密码是否正确
        try:
            check_query = """
                SELECT password, full_name FROM admins WHERE admin_id = %s
            """
            admin_info = execute_query(check_query, (admin_id,))
            
            if not admin_info:
                return {
                    "code": 404,
                    "message": "管理员信息不存在"
                }
            
            stored_password = admin_info[0]['password']
            admin_name = admin_info[0]['full_name']
            old_password_md5 = md5_password(password_data.old_password)
            
            if stored_password != old_password_md5:
                return {
                    "code": 400,
                    "message": "旧密码不正确"
                }
            
            # 更新密码
            new_password_md5 = md5_password(password_data.new_password)
            update_query = """
                UPDATE admins 
                SET password = %s, updated_at = NOW()
                WHERE admin_id = %s
            """
            affected_rows = execute_update(update_query, (new_password_md5, admin_id))
            
            if affected_rows == 0:
                return {
                    "code": 404,
                    "message": "管理员信息不存在，密码更新失败"
                }
        except Exception as e:
            # 记录错误日志
            traceback_str = traceback.format_exc()
            print(f"检查旧密码或更新密码失败: {str(e)}")
            print(f"错误详情: {traceback_str}")
            return {
                "code": 500,
                "message": f"更新密码失败: {str(e)}"
            }
        
        # 记录操作日志
        try:
            execute_update(
                """INSERT INTO operation_logs (admin_id, operation_type, operation_desc, created_at) 
                   VALUES (%s, %s, %s, NOW())""", 
                (admin_id, "更新", f"{admin_name}修改了密码")
            )
        except Exception as log_error:
            # 仅记录日志错误，不影响主流程
            print(f"记录操作日志失败: {str(log_error)}")
        
        return {
            "code": 200,
            "message": "密码修改成功"
        }
    except Exception as e:
        # 记录错误日志
        traceback_str = traceback.format_exc()
        print(f"修改管理员密码失败: {str(e)}")
        print(f"错误详情: {traceback_str}")
        # 返回错误响应
        return {
            "code": 500,
            "message": f"修改管理员密码失败: {str(e)}"
        }
