import requests
import json
import time

# 1. 配置 AK/SK 和其他参数
access_key = "YMHYKI9VXENKECP9CPYE"
secret_key = "0aMkWCOU9p42A0ESGwKDLoCTfyaGXKzY42JvvZtj"
pass_key = "Zhengai,1"
region = "cn-south-1"
project_id = "875b1a4942db47588382e0c9bc34dd46"
iamuser = "tangjianduan"
domain_name = "myj076901"
domain_id = "Password@321345"
workspace_id = "e3fcce08a90d40dfa21bc9333e1718bc"



print('start producer')




# 2. 获取 Token
def get_token():
    #    url = f"https://iam.{region}.myhuaweicloud.com/v3/auth/tokens"
    url = "https://iam.cn-south-1.myhuaweicloud.com/v3/auth/tokens"
    headers = {"Content-Type": "application/json"}
    payload = {
        "auth": {
            "identity": {
                "methods": ["password"],
                "password": {
                    "user": {
                        "name": "tangjianduan",
                        "password": "Password@321345",
                        "domain": {"name": "myj076901"},
                    }
                },
            },
            #            "scope": {"project": {"name": "cn-south-1"}},
            "scope": {"project": {"id": "875b1a4942db47588382e0c9bc34dd46"}},
        }
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    print(response.content)
    return response.headers["X-Subject-Token"]


def executeScript(script_name):
    headers = {
        "Content-Type": "application/json;charset=utf8",
        "X-Auth-Token": get_token(),
        "workspace": workspace_id,
    }

    url = f"https://dayu-dlf.cn-south-1.myhuaweicloud.com/v1/{project_id}/scripts/{script_name}/execute"
    json_body =  {}
    response = requests.post(url, headers=headers, data=json.dumps(json_body))
    instance_id = response.json()["instanceId"]
    print (instance_id)
    time.sleep(5)
    url = f"https://dayu-dlf.cn-south-1.myhuaweicloud.com/v1/{project_id}/scripts/{script_name}/instances/{instance_id}"
    json_body =  {}
    response = requests.get(url , headers=headers)
    results = response.json()["results"]
    for row in results:
        rows =row["rows"]
        print (rows)
    return None


if __name__ == '__main__':
    script_name = "tbthsplitorder"
    executeScript(script_name)
