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
import json

from layotto.client.base import AntLayottoClient
from layotto.client.request.inference import (
ContentType,
Debug,
InferenceRequest,
Item,
LayottoInferenceConfig,
MayaConfig,
MayaResponse,
TensorFeatures,
User,
convertInferenceRequestToPbRequest,
)
from layotto.core.proto import ai_pb2 as ai__pb2

if __name__ == "__main__":
    # 1. 需要初始化Mosn,这一步是对Mosn进行了初始化，这个是全局唯一的，即全局只需要应用启动时执行一次即可!!
    client = AntLayottoClient()
    client.initialize_app("iseecore")
    # 等Mosn2.19发布后跑这个ut
    # 2. 这一步是初始化Layotto的maya能力，全局执行一次即可
    client.init_inference(LayottoInferenceConfig("iseecore", "maya"))
    item = Item()
    item.set_item_id("123456789")
    tensor_features = TensorFeatures()
    string_value = \
    {
        "top_p": 0.5, 
        "trace_id": "test_trace_id", 
        "query": "你好", 
        "top_k": 0, 
        "temperature": 0, 
        "max_output_length": 30000,
        "history": "[]",
        "sync": 'true',
    }

    string_value = json.dumps(string_value)
    tensor_features.set_string_values([string_value])
    item.set_tensor_features({"data": tensor_features})

    config = MayaConfig()

    # 请求超时时间，非必填，默认600ms
    config.request_time_out = 60000

    request = InferenceRequest(
        scene_name="hekaiwen_multi_agent",
        # chain_name="241104_code_agent_tool_call_dpo",
        chain_name="241129_qwen2_5_72b_ori",
        items=[item],
        config=config,
        debug=Debug.OPEN,
    )
    # 指定预发机器的IP地址，只能做debug使用！！！！
    resp = client.inference(request,[("rpc_target_address", "11.71.111.223:10001")])
    INFO(
        f"maya call success with resp, "
        f"object_attributes: {resp.items[0].object_attributes}, "
        f"attributes: {resp.items[0].attributes}, "
        f"item_id: {resp.items[0].item_id}, "
        f"score: {resp.items[0].score}, "
        f"scores: {resp.items[0].scores}, "
        f"servers: {resp.servers}, "
        f"rt: {resp.rt}, "
        f"success: {resp.success} "
    )