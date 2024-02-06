import json
import requests
import pandas as pd

# 1. 配置 AK/SK 和其他参数
access_key = "YMHYKI9VXENKECP9CPYE"
secret_key = "0aMkWCOU9p42A0ESGwKDLoCTfyaGXKzY42JvvZtj"
pass_key = "Zhengai,1"
region = "cn-south-1"
project_id = "875b1a4942db47588382e0c9bc34dd46"
iamuser = "tangjianduan"
domain_name = "myj076901"
domain_id = "Password@321345"

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

token = get_token()
print("get token ok")

headers = {
    "Content-Type": "application/json;charset=utf8",
    "X-Auth-Token": token,
}

def update_cdm_job(json_body, source_name, headers, clusters_id):
    url = f"https://cdm.cn-south-1.myhuaweicloud.com/v1.1/875b1a4942db47588382e0c9bc34dd46/clusters/" + clusters_id + "/cdm/job/" + source_name
    response = requests.put(url, headers=headers, data=json.dumps(json_body))
    if response.status_code == 200:
        print("作业创建成功")
        return response.json()
    else:
        print(response.content)
        print("作业创建失败")
        print(taskname)
    return None
# 3. 调用 CDM 服务的 API 接口创建一个作业
def create_cdm_job(token, json_body, taskname,clusters_id):
    url = f"https://cdm.cn-south-1.myhuaweicloud.com/v1.1/875b1a4942db47588382e0c9bc34dd46/clusters/" + clusters_id + "/cdm/job"
    response = requests.post(url, headers=headers, data=json.dumps(json_body))
    if response.status_code == 200:
        print("作业创建成功")
        return response.json()
    else:
        print(response.content)
        print("作业创建失败")
        print(taskname)
    return None

# 5. 使用示例数据调用 create_cdm_job 函数
job_name = "your_job_name"
cluster_name = "your_cluster_name"

#3 读取Excel文件
df = pd.read_excel('D:\\list.xlsx')

# 读取JSON模板文件
with open('D:\\cdm_template.json', 'r') as f:
    template = f.read()
json_data = json.loads(template)

# 遍历Excel中的每一行，并将每一行的数据替换JSON模板中的关键字
for index, row in df.iterrows():
    # demo_job = row["demo_job"]
    # url = f"https://cdm.cn-south-1.myhuaweicloud.com/v1.1/875b1a4942db47588382e0c9bc34dd46/clusters/34ac6bce-5f24-49e0-8deb-bc6e256bf45b/cdm/job/"
    # response = requests.get(url + demo_job, headers=headers)
    # json_data = response.json()
    # 从 Excel 行中获取需要的字段值
    source_name = row["source_name"]
    if(source_name.find("-ADD-") != -1):
        json_data["jobs"][0]["to-config-values"]["configs"][0]["inputs"][3]["value"] = "false"
    querysql = row["sql"]
    taskname = row["task_name"]
    table_column = row["table_column"]

    target_connection_name = row["target_connection_name"]
    target_database_name = row["target_database_name"]
    target_table_name = row["target_table_name"]
    target_table_column = row["target_table_column"]

    group_name = row["group_name"]
    group_id = row["group_id"]

    # ... 以此类推，根据需要添加更多字段

    # 替换 JSON 模板中的占位符
    json_data["jobs"][0]["from-link-name"] = taskname
    json_data["jobs"][0]["name"] = source_name
    json_data["jobs"][0]["from-config-values"]["configs"][0]["inputs"][2]["value"] = querysql
    json_data["jobs"][0]["from-config-values"]["configs"][0]["inputs"][2]["size"] = 5000

    json_data["jobs"][0]["from-config-values"]["configs"][0]["inputs"][3]["value"] = table_column
    json_data["jobs"][0]["from-config-values"]["configs"][0]["inputs"][3]["size"] = 5555

    json_data["jobs"][0]["to-link-name"] = target_connection_name
    json_data["jobs"][0]["to-config-values"]["configs"][0]["inputs"][0]["value"] = target_database_name
    json_data["jobs"][0]["to-config-values"]["configs"][0]["inputs"][1]["value"] = target_table_name
    json_data["jobs"][0]["to-config-values"]["configs"][0]["inputs"][2]["value"] = target_table_column

    json_data["jobs"][0]["driver-config-values"]["configs"][6]["inputs"][1]["value"] = group_name
    json_data["jobs"][0]["driver-config-values"]["configs"][6]["inputs"][0]["value"] = str(group_id)

    index = str(row["index"])
    cdm2 = "34ac6bce-5f24-49e0-8deb-bc6e256bf45b"
    cdm3 = "341abd0a-b185-462a-b096-4ea70676a200"
    if index.isdigit():
        create_cdm_job(token, json_data, taskname, cdm3)
    else:
        update_cdm_job(json_data, source_name, headers, cdm3)


#result = create_cdm_job(token, json_body)
#print(result)


