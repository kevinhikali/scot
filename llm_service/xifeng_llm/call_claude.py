# coding: utf-8
# Copyright (c) Antfin, Inc. All rights reserved.
# 通过枢纽访问GPT各模型，详细内容参考 https://yuque.antfin-inc.com/sjsn/biz_interface/pdxkcxwic4kdarrf
#
import os
import json
import requests
import csv
import ast
import html
from Crypto.Cipher import AES
from binascii import b2a_hex, a2b_hex
import time
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    before_sleep
)
import traceback
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
from loguru import logger

AVAILABLE_MODEL_LIST = ['gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-4', 'Doubao-pro-128k', 'claude-3.5-sonnet']

class ZdfmngLLM(object):
    '''
        该类封装了通过枢纽访问GPT的模式
    '''
    name = 'zdfmg_llm'

    zdfmng_url = 'https://zdfmng.alipay.com/commonQuery/queryData'

    verbose: bool = False

    max_n_retry = 10
    min_retry_seconds = 5
    max_retry_seconds = 10

    stream: bool = False

    def __init__(self, cfg: Dict, **kwargs):
        self.cfg = cfg
        self.model = self.cfg.get('model', AVAILABLE_MODEL_LIST[0])
        assert self.check_model(self.model)

        self.model_id = self.model

        self.zdfmng_url = self.cfg.get('zdfmng_url', self.zdfmng_url)
        self.verbose = kwargs.get('verbose', self.verbose)
        self.max_n_retry = kwargs.get('max_n_retry', self.max_n_retry)
        self.min_retry_seconds = kwargs.get('min_retry_seconds', self.min_retry_seconds)
        self.max_retry_seconds = kwargs.get('max_retry_seconds', self.max_retry_seconds)

        # Load environment variables
        # the following variables are required, if not exist, will raise KeyError Exception
        self.AES_KEY = os.environ["AES_KEY"]
        # self.GPT35_VISIT_DOMAIN = os.environ['GPT35_VISIT_DOMAIN']
        # self.GPT35_VISIT_BIZ = os.environ['GPT35_VISIT_BIZ']
        # self.GPT35_VISIT_BIZLINE = os.environ["GPT35_VISIT_BIZLINE"]
        # self.GPT4_VISIT_DOMAIN = os.environ["GPT4_VISIT_DOMAIN"]
        # self.GPT4_VISIT_BIZ = os.environ["GPT4_VISIT_BIZ"]
        # self.GPT4_VISIT_BIZLINE = os.environ["GPT4_VISIT_BIZLINE"]
        self.API_KEY = os.environ["API_KEY"]
        self.SCENE_CODE = os.environ['SCENE_CODE']

        self.CLAUDE_VISIT_DOMAIN = os.environ['CLAUDE_VISIT_DOMAIN']
        self.CLAUDE_VISIT_BIZ = os.environ['CLAUDE_VISIT_BIZ']
        self.CLAUDE_VISIT_BIZLINE = os.environ["CLAUDE_VISIT_BIZLINE"]

        # self.DOUBAO_VISIT_DOMAIN = os.environ['DOUBAO_VISIT_DOMAIN']
        # self.DOUBAO_VISIT_BIZ = os.environ['DOUBAO_VISIT_BIZ']
        # self.DOUBAO_VISIT_BIZLINE = os.environ["DOUBAO_VISIT_BIZLINE"]
        # self.USER_NAME = os.environ.get('USER_NAME', 'default')

    def generate(self,
                 llm_artifacts: Union[str, List[dict]],
                 functions: List = [],
                 **kwargs) -> Any:
        '''
        Returns:
            message: {'role': 'xx', 'content': xx, 'function_call': xx}
        '''

        params_dict = self.build_request_params(**kwargs)

        if isinstance(llm_artifacts, str):
            llm_artifacts = [{"role": "user", "content": llm_artifacts}]

        params_dict["queryConditions"]["messages"] = llm_artifacts

        if len(functions):
            params_dict['queryConditions']['function_call'] = 'auto'
            params_dict['queryConditions']['functions'] = functions

        if self.verbose:
            # logger.info(f"zdfmng request: {json.dumps(params_dict['queryConditions'], ensure_ascii=False)}") #json.dumps(params_dict, ensure_ascii=False, indent=2))
            log_data = {'zdfmng_request': json.dumps(params_dict['queryConditions'], ensure_ascii=False)}
            log_data = json.dumps(log_data, ensure_ascii=False)
            logger.info(f"zdfmng request: {log_data}")

        # call zdfmng
        max_n_retry = kwargs.get('max_n_retry', self.max_n_retry)
        min_retry_seconds = kwargs.get('min_retry_seconds', self.min_retry_seconds)
        max_retry_seconds = kwargs.get('max_retry_seconds', self.max_retry_seconds)
        response = self.call_zdfmng_with_retry(params_dict, max_n_retry, min_retry_seconds, max_retry_seconds)

        return response

    def stream_generate(self,
                        llm_artifacts: Union[str, List[dict]],
                        functions: List = [],
                        **kwargs) -> Any:
        # TODO 枢纽不支持stream模式，用generate模式来替代
        raise NotImplementedError("zdfmng_llm do not support stream_generate() ")

    def build_request_params(self, **kwargs: Any) -> Any:
        if self.model in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k"]:
            VISIT_DOMAIN = self.GPT35_VISIT_DOMAIN
            VISIT_BIZ = self.GPT35_VISIT_BIZ
            VISIT_BIZLINE = self.GPT35_VISIT_BIZLINE
        elif self.model in ["gpt-4", "gpt-4-32k", 'gpt-4-turbo']:
            VISIT_DOMAIN = self.GPT4_VISIT_DOMAIN
            VISIT_BIZ = self.GPT4_VISIT_BIZ
            VISIT_BIZLINE = self.GPT4_VISIT_BIZLINE
        elif self.model in ["Doubao-pro-128k"]:
            VISIT_DOMAIN = self.DOUBAO_VISIT_DOMAIN
            VISIT_BIZ = self.DOUBAO_VISIT_BIZ
            VISIT_BIZLINE = self.DOUBAO_VISIT_BIZLINE
        elif self.model in ["claude-3.5-sonnet"]:
            VISIT_DOMAIN = self.CLAUDE_VISIT_DOMAIN
            VISIT_BIZ = self.CLAUDE_VISIT_BIZ
            VISIT_BIZLINE = self.CLAUDE_VISIT_BIZLINE
        else:
            assert False, "Unknown model"

        if self.model in ["Doubao-pro-128k"]:
            param_dict = {
                "serviceName": "doubao_chat_completions_dataview",
                "visitDomain": VISIT_DOMAIN,
                "visitBiz": VISIT_BIZ,
                "visitBizLine": VISIT_BIZLINE,
                "cacheInterval": -1,  # 不缓存
                "queryConditions": {
                    "model": self.model,
                    "messages": [],  # placeholder for prompt
                    "temperature": "0.8",
                    "max_tokens": 4096,
                }
            }
        elif self.model in ["claude-3.5-sonnet"]:
            param_dict = {
                "serviceName": "amazon_claude_chat_completions_dataview",
                "visitDomain": VISIT_DOMAIN,
                "visitBiz": VISIT_BIZ,
                "visitBizLine": VISIT_BIZLINE,
                "cacheInterval": -1,  # 不缓存
                "queryConditions": {
                    "model": self.model,
                    "messages": [],  # placeholder for prompt
                    "temperature": "0.8",
                    "max_tokens": 4096,
                }
            }
        else:
            param_dict = {
                "serviceName": "chatgpt_prompts_completions_query_dataview",
                "visitDomain": VISIT_DOMAIN,
                "visitBiz": VISIT_BIZ,
                "visitBizLine": VISIT_BIZLINE,
                "cacheInterval": -1,  # 不缓存
                "queryConditions": {
                    "requestName": self.USER_NAME,
                    "model": self.model,
                    "api_key": self.API_KEY,
                    "messages": [],  # placeholder for prompt
                    "scene_code": self.SCENE_CODE,
                    "temperature": "0.8",
                    "stream": False,
                    "max_tokens": 4096,  # 1、gpt-3.5-turbo模型，支持4,096 token 2、gpt-3.5-turbo-16k模型，支持16,384 token
                }
            }

        # udpate model parameters by user-defined, e.g: temperature, max_tokens, stop
        for k, v in kwargs.items():
            if k not in ['scene_code', 'api_key', 'requestName', 'model', 'messages']:
                param_dict['queryConditions'][k] = v

        return param_dict

    @staticmethod
    def check_model(model: str) -> bool:
        if model and model in AVAILABLE_MODEL_LIST:
            return True
        return False

    def call_zdfmng_with_retry(self,
                               params: Dict,
                               max_n_retry: int = 3,
                               min_retry_seconds: int = 1,
                               max_retry_seconds: int = 2
                               ) -> Any:
        '''
            A retry wrapper of call_zdfmng()
        '''

        @retry(reraise=True,
               stop=stop_after_attempt(max_n_retry),
               wait=wait_exponential(multiplier=1,
                                     min=min_retry_seconds,
                                     max=max_retry_seconds)
               )
        def _call_zdfmng_with_retry(params: Dict) -> Any:
            return self.call_zdfmng(params)

        return _call_zdfmng_with_retry(params)

    def call_zdfmng(self, params: Dict) -> Any:
        data = json.dumps(params)
        encrypted_data = self.aes_encrypt(data)

        post_data = {"encryptedParam": encrypted_data}
        headers = {'Content-Type': 'application/json'}

        try:
            response = requests.post(self.zdfmng_url,
                                     data=json.dumps(post_data),
                                     headers=headers)
            ast_str = ast.literal_eval("'" + response.json()["data"]["values"]["data"] + "'")
            js = html.unescape(ast_str)
            data = json.loads(js)
            if self.verbose:
                log_data = {'zdfmng_response': json.dumps(data, ensure_ascii=False)}
                log_data = json.dumps(log_data, ensure_ascii=False)
                logger.info(f"zdfmng response: {log_data}")

            message = data["choices"][0]["message"]
            prompt_tokens = data["usage"]["prompt_tokens"]
            total_tokens = data["usage"]["total_tokens"]
            return message

        except Exception as e:
            error = traceback.format_exc()

            status_code, return_text = None, None
            if isinstance(response, requests.Response):
                status_code = response.status_code
                return_text = response.text

            logger.error(f'call zdfmng error, status_code: {status_code}, response: {return_text}, traceback: {error}')
            raise RuntimeError(error)

        return None

    def aes_encrypt(self, data: str) -> str:
        '''
        Args:
            data: string, whic is the result of json.dumps(http_request_param)

        Returns:
            encrypt_result: string, data ecrypted by AES using self.AES_KEY
        '''
        iv = "1234567890123456"
        cipher = AES.new(self.AES_KEY.encode('utf-8'), AES.MODE_CBC, iv.encode('utf-8'))  # 设置AES加密模式 此处设置为CBC模式
        block_size = AES.block_size

        # 判断data是不是16的倍数，如果不是用b'\0'补足
        if len(data) % block_size != 0:
            add = block_size - (len(data) % block_size)
        else:
            add = 0
        data = data.encode('utf-8') + b'\0' * add
        encrypted = cipher.encrypt(data)  # aes加密
        result = b2a_hex(encrypted)  # b2a_hex encode  将二进制转换成16进制
        return result.decode('utf-8')

if __name__ == '__main__':
    response = batch_claude_predict(prompt_list=['介绍下杭州西湖，不超过 100 字'] * 5)
    print(response)
    

