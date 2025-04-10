import time
import subprocess
import os
from datetime import datetime, timedelta

def main():
    # 计算3小时后的时间
    target_time = datetime.now() + timedelta(hours=3)
    print(f"将在 {target_time.strftime('%Y-%m-%d %H:%M:%S')} 开始执行")
    
    # 等待3小时
    time.sleep(3 * 60 * 60)
    
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, "produce_questions.py")
    
    # 执行脚本
    print(f"开始执行脚本: {script_path}")
    try:
        subprocess.run(["python", script_path], check=True)
        print("脚本执行完成")
    except subprocess.CalledProcessError as e:
        print(f"脚本执行失败: {str(e)}")

if __name__ == "__main__":
    main() 