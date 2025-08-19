import os
import sys
import requests
parent_dir = os.path.dirname(os.path.abspath(__file__))
up_dir = parent_dir
for i in range(3):
    sys.path.append(up_dir)
    up_dir = os.path.dirname(up_dir)
from kutils import DEBUG, INFO, WARN, ERROR
import utils as u
from maya_http_requestor import MayaHttpRequestor

if __name__ == "__main__":

    # 预发 paiplusinferencepre
    # 生产 paiplusinference

    # 继续问
    params = {
        "url_pre" : "https://paiplusinferencepre.alipay.com/inference/9d716ad116753ef7_hekaiwen_continuous_ask/241104",
        "url_prod" : "https://paiplusinference.alipay.com/inference/9d716ad116753ef7_hekaiwen_continuous_ask/241104",
    }

    # CodeAgent
    # params = {
    #     "url_pre" : "https://paiplusinferencepre.alipay.com/inference/882e000f7e28fdb1_hekaiwen_multi_agent/241104_code_agent_tool_call_dpo",
    #     "url_prod" : "https://paiplusinference.alipay.com/inference/882e000f7e28fdb1_hekaiwen_multi_agent/241104_code_agent_tool_call_dpo",
    # }

    # params = {
    #     "url_pre" : "https://paiplusinferencepre.alipay.com/inference/882e000f7e28fdb1_hekaiwen_multi_agent/241129_qwen2_5_72b_ori",
    #     "url_prod" : "https://paiplusinference.alipay.com/inference/882e000f7e28fdb1_hekaiwen_multi_agent/241129_qwen2_5_72b_ori",
    # }

    query = "你是谁，你是一个多大的模型？"
    history = None
    mhr = MayaHttpRequestor(params["url_pre"])
    result = mhr.process(query, history)
    DEBUG(result)
    DEBUG(result['resultMap']['result'])