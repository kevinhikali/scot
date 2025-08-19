import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
up_dir = parent_dir
for i in range(3):
    sys.path.append(up_dir)
    up_dir = os.path.dirname(up_dir)
from kutils import DEBUG, INFO, WARN, ERROR
import utils as u
import re
import ast
import json
import sys
from string import Template
from multiprocessing import Process, Pool
import os
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import datetime
import regex
import json
from typing import List, Dict, Union, Tuple, Any
from abc import ABC, abstractmethod
from llm_service.xifeng_llm.call_chatgpt import ChatGPT
from copy import deepcopy
import traceback
from typing import Any
import requests
import json
import re
from openai import OpenAI
from pyhocon import ConfigFactory


"""
LLM工厂类，统一各类模型的访问方式，支持

gpt-4-turbo
gpt3.5
gpt4
claude
claude-v2
qwen_2_72b_chat
Bailing-4.0-10B-16K-Chat
Bailing-10b-flight-agent -> 机票Agent的线上模型

"""


class LLMModel(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __call__(self, messages, **kwargs):
        raise NotImplementedError


class GPTModel(LLMModel):
    """
    封装了如下模型的访问方式
    gpt-4-turbo
    gpt3.5
    gpt4
    claude
    claude-v2
    """
    def __init__(self, model_name, config_path):
        self.model_name = model_name
        self.client = ChatGPT(config_path)

    def __call__(self, messages, **kwargs):
        try:
            self.client.update_conf(kwargs)
            resp = self.client.predict(messages)
            self.client.reset_conf()
            return resp
        except Exception as e:
            print(traceback.print_exc())
            return None

class DeepSeekModel(LLMModel):
    """
    封装了如下模型的访问方式
    deepseek_r1
    """

    def __init__(self, model_name, config_path):
        self.model_name = model_name

        self.config_path = config_path

        self.config = ConfigFactory.parse_file(config_path)

        self.client = OpenAI(
            api_key=self.config.get("api_key"),
            base_url=self.config.get("base_url")
        )

    def __call__(self, messages, **kwargs):
        try:
            completion = self.client.chat.completions.create(
                model=self.config.get("model"),
                messages=messages,
                **kwargs
            )
            resp = completion.choices[0].message.content
            # 如果是o1类型的模型，需要把思考过程去除掉
            if self.config.get("llm_type") == "o1":
                resp = re.sub('\<think\>[\w\W]+?\<\/think\>', '', resp).strip()

            return resp
        except Exception as e:
            return None

class AliModel(LLMModel):
    """
    封装了如下模型的访问方式
    qwen_2_72b_chat
    Bailing-4.0-10B-16K-Chat
    """

    def __init__(self, model_name):
        self.model_name = model_name

    def __call__(self, messages, **kwargs):
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer xi00r4DELIA3uqWH159lHGhBgKZZ1ras",
        }
        response = requests.post(
            'https://antchat.alipay.com/v1/chat/completions',
            headers=headers,
            json={"model": self.model_name, "messages": messages}
        )
        try:
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            return None

class ZhuliLingxiModel(LLMModel):
    def __init__(self, model_name, config_path):
        self.model_name = model_name
        self.config_path = config_path

        self.config = ConfigFactory.parse_file(config_path)
    
    def __call__(self, messages, **kwargs):
        headers = {
            'Content-Type': 'application/json;charset=UTF-8',
            'Authorization': 'Bearer zhuli',
        }

        req_data = dict(self.config.get('req_template'))
        req_data['query'] = messages
        
        try:
            url = self.config.get('pre_host') if self.config.get('url_type') is None else self.config.get('prod_host')
            response = requests.post(url,
                            headers=headers,
                            data=json.dumps(req_data),
                            timeout=60)

            return response.json()['response']
        except Exception as e:
            print(f"post exception: {e}")

        return ""

class LLMFactory:
    def __init__(self):
        self.config = {
            "gpt-4-turbo": "config/ChatGPT_alps.conf",
            "gpt3.5": "config/ChatGPT_alps_gpt3_5.conf",
            "gpt4": "config/ChatGPT_alps_gpt4o.conf",
            "claude": "config/Claude.conf",
            "claude-v2": "config/Claude-v2.conf",
            "qwen_2_72b_chat": "", 
            "Bailing-4.0-10B-16K-Chat": "",
            "Bailing-10b-flight-agent": "config/Bailing-10b-flight-agent.conf",
            "deepseek_r1": "config/deepseek_r1.conf"
        }

    def create_llm(self, model_name: str):
        if model_name not in self.config: return None
        config_path = self.config[model_name]
        dir_path = os.path.dirname(os.path.realpath(__file__))
        config_file = f'{dir_path}/{config_path}'
        model_config = ConfigFactory.parse_file(config_file)
        model_type = model_config.get('model_type')

        try:
            model_cls = eval(f'{model_type}Model')
            return model_cls(model_name, config_file)
        except Exception as e:
            print(e)
            return AliModel(model_name)

def task_prompt_wrapper(prompt):
    return [{'role': 'user', 'content': prompt}]

def llm_call_with_retry(llm, prompt, num_retry=3, new_param_conf={}):
    retry = 0
    while retry < num_retry:
        retry += 1

        gen_resp = llm(task_prompt_wrapper(prompt), **new_param_conf)

        if gen_resp is not None:
            return gen_resp
    return None

def parse_func_name(func_expression):
        try:
            return regex.findall('.+?(?=\()', func_expression)[0]
        except Exception as e:
            return None

def parse_func_args(func_expression):
    try:
        args_str = regex.findall('(?<=\().+?(?=\))', func_expression)[0]
        args_dict = dict()

        quotation = '"'

        if "='" in args_str:
            quotation = "'"

        args_arr = args_str.split(f'{quotation},')
        for idx, item in enumerate(args_arr):
            if idx == len(args_arr) - 1:
                item = item[:-1]
            arg_name, arg_val = item.split(f'={quotation}')
            args_dict[arg_name.strip()] = arg_val.strip()
        return args_dict
    except Exception as e:
        print(e)
        return None

def parse_function_expression(func_expression):
    func_name = parse_func_name(func_expression)
    args_dict = parse_func_args(func_expression)
    return func_name, args_dict

def if_contains_chinese(message):
    matched = regex.findall('[\u4e00-\u9fa5]', message)
    return len(matched) > 0

def load_data(file_path: str) -> List[Dict]:
    data_dict_list = []
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            data_dict_list.append(json.loads(line.strip()))
    
    return data_dict_list

def save_data(data_dict_list: List[Dict], file_path: str) -> None:
    with open(file_path, 'w', encoding='utf8') as f:
        for data_dict in data_dict_list:
            f.write(json.dumps(data_dict, ensure_ascii=False) + '\n')

def request_xifeng_llm(messages: dict):
    llm = LLMFactory().create_llm("gpt-4-turbo")
    response = llm(messages)
    return response

# def request_xifeng_llm(prompt: str):
#     llm = LLMFactory().create_llm("gpt-4-turbo")
#     response = llm_call_with_retry(llm, prompt)
#     return response

if __name__ == '__main__':
    llm = LLMFactory().create_llm('gpt-4-turbo')
    messages = [
        {"role": "user", "content": "你好"},
    ]
    resp = llm(messages)
    print(resp)

