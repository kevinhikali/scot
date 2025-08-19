import os
import json
import requests
import time
from fastapi import HTTPException

CONFIG = {
    "model": "qwen2-72b-chat",
    "model_client_cls": "XYModelClient",
    "device": "cpu",
    "lingxi_token": "Bearer ZF6SsxdZRuJhcmVRZ5dDBcJe0tKuD8lu",
    "uuid": "2088802552554538",
    "n": 1,
    "params": {
        "temperature": "0.0",
        "doSample": "false",
        "max_output_length": "1024"
    },
    "model_id": "qwen_2_72b_chat",
    "version": "v1",
    "options": {
        "temperature": "0.0",
        "doSample": "false",
        "max_output_length": "1024"
    }
}

class LingxiClient(object):
    def __init__(self, model):
        self.url = 'https://aichat-pre.alipay.com/api/v1/completion/chat'
        self.model_id = model
        self.version = CONFIG['version']
        self.header = {"Content-Type": "application/json", "Authorization": "Bearer ZF6SsxdZRuJhcmVRZ5dDBcJe0tKuD8lu"}
        self.uuid = '2088002255969523'

    def generate(self, messages, **kwargs):
        if isinstance(messages, list):
            prompt = '\n\n'.join([f"{msg['role'].upper()}\n{msg['content']}" for msg in messages])
        else:
            prompt = messages

        if 'max_output_length' in kwargs:
            max_output_length = int(kwargs.get('max_output_length'))
        else:
            max_output_length = 500
        req_data = {
            'query': prompt,
            "userId": self.uuid,
            "reqType": "chat",
            "extraParams": {
            },
            "modelOption": {
                "modelId": self.model_id,
                "version": self.version
            },
        }
        max_retry = int(kwargs.get('max_retry')) if 'max_retry' in kwargs else 3
        retry = 0
        while (retry < max_retry):
            try:
                result = self._call_lingxi(req_data)
                return result
            except Exception as ex:
                print(f"call maya failed. error: {ex}")
                retry += 1
                time.sleep(3)

        return ''

    def _call_lingxi(self, req_data, timeout=60):
        response = requests.post(self.url,
                                 json=req_data,
                                 headers=self.header,
                                 timeout=timeout,
                                 )
        if 200 != response.status_code:
            print(f"{response.status_code} {response.text}")
            raise HTTPException(response.status_code, "Call maya error.")

        json_content = json.loads(response.text)
        try:
            if not json_content['success']:
                print(f"lingxi return not ok. {json_content}")
                raise KeyError("success is not true")
            content = json_content["data"]["content"]
        except KeyError:
            print(f"Error encountered: {json_content}")
            return ""
        return content

def request_qwen(prompt, model):
    qwen = LingxiClient(model)
    result = qwen.generate(prompt)
    return result
