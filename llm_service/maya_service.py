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

class MayaClientService():
    def __init__(self, scene_name, chain_name, timeout=60000, mode='pre'):
        self.scene_name = scene_name
        self.chain_name = chain_name
        self.mode = mode
        self.client = AntLayottoClient()
        self.client.initialize_app("iseecore")
        self.client.init_inference(LayottoInferenceConfig("iseecore", "maya"))
        self.config = MayaConfig()
        self.config.request_time_out = timeout

    def call(self, query):
        item = Item()
        item.set_item_id("123456789")
        request = InferenceRequest(
            scene_name = self.scene_name,
            chain_name = self.chain_name,
            items=[item],
            config=self.config,
            debug=Debug.OPEN,
        )
        tensor_features = TensorFeatures()
        string_value = \
        {
            "top_p": 0.5, 
            "trace_id": "test_trace_id", 
            "query": query, 
            "top_k": 0, 
            "temperature": 0.3, 
            "max_output_length": 2000,
            "history": "[]",
            "sync": 'true',
        }
        string_value = json.dumps(string_value)
        tensor_features.set_string_values([string_value])
        item.set_tensor_features({"data": tensor_features})
        resp = None
        if self.mode == 'pre':
            resp = self.client.inference(request, [("rpc_target_address", "11.71.111.223:10001")])
        else:
            resp = self.client.inference(request)
        result = resp.items[0].attributes['result']
        return result 

def request_maya_client(prompt, model):
    scene_name = "hekaiwen_multi_agent"
    chain_name = "241104_code_agent_tool_call_dpo"
    mcs = MayaClientService(scene_name, chain_name)
    result = mcs.call(prompt)
    return result

if __name__ == "__main__":
    scene_name = "hekaiwen_multi_agent"
    chain_name = "241104_code_agent_tool_call_dpo"
    mcs = MayaClientService(scene_name, chain_name)
    result = mcs.call("你在哪")
    DEBUG(result)
