import requests

def send_sms_report_request():
    """
    发送短信报告请求到国阳云API
    """
    # API请求URL
    url = "http://api.guoyangyun.com/api/sms/smsReport.htm"
    
    # 请求参数
    params = {
        "appkey": "948234057202501301243564509412",
        "appsecret": "cfd37613dff29f0fed8bd5a061ddff3d",
        "smsid":"17439300359159482340572594"
    }
    
    try:
        # 发送GET请求
        response = requests.get(url, params=params)
        
        # 检查请求是否成功
        response.raise_for_status()
        
        # 打印响应内容
        print("状态码:", response.status_code)
        print("响应内容:", response.text)
        
        return response.json() if response.headers.get('content-type') == 'application/json' else response.text
    
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return None

if __name__ == "__main__":
    # 执行请求
    result = send_sms_report_request()
