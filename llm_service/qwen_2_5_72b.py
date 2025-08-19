import json
import logging

# -*- coding: utf-8 -*-

import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
up_dir = parent_dir
for i in range(3):
    sys.path.append(up_dir)
    up_dir = os.path.dirname(up_dir)
from kutils import DEBUG, INFO, WARN, ERROR
import utils as u

from layotto.client.base import AntLayottoClient
from layotto.client.request.inference import (
    Debug,
    InferenceRequest,
    Item,
    LayottoInferenceConfig,
    MayaConfig,
    TensorFeatures,
)

class QwenClient():
    def __init__(self, timeout=60000):
        self.client = AntLayottoClient()
        self.client.initialize_app("iseecore")
        self.client.init_inference(LayottoInferenceConfig("iseecore", "maya"))
        self.config = MayaConfig()
        self.config.request_time_out = timeout
    
    def request(self, query):
        item = Item()
        item.set_item_id("123456789")
        tensor_features = TensorFeatures()
        tensor_features.set_string_values(
            [
                json.dumps(
                    {
                        "__entry_point__": "openai.chat.completion",
                        "model": "auto",
                        "messages": [
                            {"role": "user", "content": query},
                        ],
                        "stream": False,
                        # 以下5个推理参数会影响推理速度和效果，如非必要可不填
                        "max_tokens": 30000,
                        "temperature": 0,
                        "repetition_penalty": 1.09,
                        "top_p": 0.95,
                        "top_k": 20,
                    }
                )
            ]
        )
        item.set_tensor_features({"data": tensor_features})

        request = InferenceRequest(
            scene_name="Qwen2_72B_Instruct_vllm",
            chain_name="v1",
            items=[item],
            config=self.config,
            debug=Debug.OPEN,
        )
        result = ''
        for resp in self.client.stream_inference(request):
            # logger.info(
            #     f"maya stream call success with resp, "
            #     f"object_attributes: {resp.items[0].object_attributes}, "
            #     f"attributes: {resp.items[0].attributes}, "
            #     f"item_id: {resp.items[0].item_id}, "
            #     f"score: {resp.items[0].score}, "
            #     f"scores: {resp.items[0].scores}, "
            #     f"servers: {resp.servers}, "
            #     f"rt: {resp.rt}, "
            #     f"success: {resp.success} "
            # )
            result = json.loads(resp.items[0].attributes['resp'])['choices'][0]['message']
            break
        return result

if __name__ == "__main__":
    qc = QwenClient()
    result = qc.request('给我讲个复杂点的故事')
    DEBUG(result)
    