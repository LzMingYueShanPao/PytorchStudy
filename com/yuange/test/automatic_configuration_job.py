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
workspace_id = "e3fcce08a90d40dfa21bc9333e1718bc"


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

# 创建CDM作业
def create_cdm_job(json_body, job_name):
    url = f"https://dayu-dlf.cn-south-1.myhuaweicloud.com/v1/875b1a4942db47588382e0c9bc34dd46/jobs"
    response = requests.post(url, headers=headers, data=json.dumps(json_body))
    if response.status_code == 204:
        url = f"https://dayu-dlf.cn-south-1.myhuaweicloud.com/v1/875b1a4942db47588382e0c9bc34dd46/jobs/"+job_name+"/start"
        requests.post(url, headers=headers, data=json.dumps(json_body))
        print("更新作业创建成功")
        # return response.json()
    else:
        print(response.content)
        print("更新创建失败")
    return None


def update_cdm_job(json_body, job_name):
    url = f"https://dayu-dlf.cn-south-1.myhuaweicloud.com/v1/875b1a4942db47588382e0c9bc34dd46/jobs/" + job_name
    response = requests.put(url, headers=headers, data=json.dumps(json_body))
    if response.status_code == 204:

        print("更新作业创建成功")
    else:
        print(response.content)
        print("更新创建失败")
    return None


df = pd.read_excel('D:\\job.xlsx')
headers = {
    "Content-Type": "application/json;charset=utf8",
    "X-Auth-Token": token,
    "workspace": workspace_id,
}
for index, row in df.iterrows():
    source_name = row["source_name"]
    job_name = row["job_name"].strip()
    demo_job =  row["job_name"]
    if str(source_name) != 'nan':
        demo_job = row["demo_job"]

    url = f"https://dayu-dlf.cn-south-1.myhuaweicloud.com/v1/875b1a4942db47588382e0c9bc34dd46/jobs/"
    response = requests.get(url + demo_job, headers=headers)
    result = response.json()
    retryTimes = result["nodes"][0]["retryTimes"]
    if (retryTimes == 0):
        result["nodes"][0]["retryTimes"] = 5
    try:
        retryTimes = result["nodes"][1]["retryTimes"]
        if (retryTimes == 0):
            result["nodes"][1]["retryTimes"] = 5
    except Exception as _:
        print("path is not exist")
    print (result)
    lst = job_name.split(".")
    # print (demo_job +" "+ result["params"][0]["value"] +result["params"][1]["value"])
    # result["params"][0]["value"] = "${yyyyMMdd-15}"
    # result["params"][1]["value"] = "${yyyyMMdd+1}"
    # result["params"][2]["value"] = "${yyyyMMdd-15}"
    # result["params"][3]["value"] = "${yyyyMMdd+1}"

    # result["directory"] = row["directory"]
    # result["schedule"]["cron"]["dependJobs"]["jobs"] = 'all_job_day1040_hxzzwms'
    # result["schedule"]["cron"]["dependJobs"]["sameWorkSpaceJobs"][0]["jobName"] = 'all_job_day1040_hxzzwms'
    # result["basicConfig"]["owner"] = "tangjianduan"
    # result["basicConfig"]["priority"] = "2"
    if str(source_name) == 'nan':
        update_cdm_job(result, job_name)
    else:
        result["lastUpdateUser"] = "tangjianduan"
        result["name"] = job_name
        result["directory"] = row["directory"]
        type = row["type"]
        if type == 'add':
            result["nodes"][1]["name"] = source_name
            result["nodes"][1]["properties"][3]["value"] = source_name
            result["nodes"][0]["properties"][6]["value"] = "enddate=\nbegdate=\nschema_name=" + lst[0] +"\ntable_name=" + lst[1] +"\nfield_name=updatetime"
            # result["nodes"][0]["name"] = "sdi_delete_table_where_begdate_and_enddate_common_p"
            # result["nodes"][0]["preNodeName"] = "sdi_delete_table_where_begdate_and_enddate_common_p"
            # result["nodes"][1]["preNodeName"] = "sdi_delete_table_where_begdate_and_enddate_common_p"
        else:
            result["nodes"][0]["name"] = source_name
            result["nodes"][0]["properties"][3]["value"] = source_name

        #
        create_cdm_job(result, job_name)

# result = create_cdm_job(token, json_body)
# print(result)
