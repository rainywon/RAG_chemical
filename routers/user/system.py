from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from datetime import datetime
from database import execute_query

router = APIRouter()

@router.get("/system/status")
async def get_system_status() -> Dict[str, Any]:
    """
    获取系统状态信息
    """
    try:
        # 获取当前版本信息
        version_info = execute_query("""
            SELECT version_number, knowledge_base_version, release_date 
            FROM system_versions 
            WHERE is_current = 1
        """)
        
        if not version_info:
            raise HTTPException(status_code=404, detail="未找到当前版本信息")
            
        version_info = version_info[0]  # 获取第一条记录
        
        # 获取系统配置信息
        configs = execute_query("""
            SELECT config_key, config_value 
            FROM system_configs 
            WHERE config_key IN ('data_sources_count', 'response_time_limit', 'system_status')
        """)
        
        # 转换为字典格式
        config_dict = {item['config_key']: item['config_value'] for item in configs}
        
        # 获取系统运行状态
        system_status = config_dict.get('system_status', 'normal')
        
        return {
            "knowledge_base": version_info['knowledge_base_version'],
            "update_date": version_info['release_date'].strftime("%Y-%m-%d"),
            "data_sources": f"{config_dict.get('data_sources_count', '0')}个专业库",
            "response_time": f"≤ {config_dict.get('response_time_limit', '3')}秒",
            "system_status": system_status
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"系统错误: {str(e)}") 