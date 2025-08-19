# coding: utf-8
# Copyright (c) Antfin, Inc. All rights reserved.

from typing import List, Dict
from Crypto.Cipher import AES
from binascii import b2a_hex, a2b_hex
import requests
import json
from pyhocon import ConfigFactory
from copy import deepcopy
import html
import ast
import time

from log_config import logger
from call_llm import Inference


def aes_encrypt(data, key="gs540iivzezmidi3"):
    """aes加密函数，如果data不是16的倍数【加密文本data必须为16的倍数！】，那就补足为16的倍数
    :param key:
    :param data:
    """
    iv = "1234567890123456"
    cipher = AES.new(key.encode('utf-8'), AES.MODE_CBC, iv.encode('utf-8'))  # 设置AES加密模式 此处设置为CBC模式
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


def aes_decode(data, key="gs540iivzezmidi3"):
    """aes解密
    :param key:
    :param data:
    """
    iv = '1234567890123456'
    cipher = AES.new(key.encode('utf-8'), AES.MODE_CBC, iv.encode('utf-8'))
    result2 = a2b_hex(data)  # 十六进制还原成二进制
    decrypted = cipher.decrypt(result2)
    return decrypted.rstrip(b'\0')  # 解密完成后将加密时添加的多余字符'\0'删除


def get_key_pairs(mode):
    seq_dict = DEV_VISIT
    if mode == "online":
        seq_dict = ONLINE_VISIT
    elif mode == "online_gpt4":
        seq_dict = ONLINE_GPT4_VISIT
    return seq_dict


def build_req_param(user, str_id, messages, mode="dev"):
    """
    str_id: 将作为key，用于异步查询结果
    messages: 文本内容，一个例子："[{\"role\":\"system\",\"content\":\"你是一个关于知识图谱行业的助手\"},{\"role\":\"user\",\"content\":\"帮我介绍一下常用的数据库类型\"}]"
    mode: dev or online
    """
    param = {
        "serviceName": "asyn_chatgpt_prompts_completions_query_dataview",
        "cacheInterval": "-1",
        "queryConditions": {
            # "model": "gpt-3.5-turbo",
            "model": "gpt-4-turbo",
            "messages": messages,
            "max_tokens": "150",
            "temperature": "1",
            "n": "1",
            "top_p": "1",
            "outputType": "PULL",
            "messageKey": f"{user}_{str_id}",
            "api_key": API_KEY,
        }
    }

    param.update(get_key_pairs(mode))
    return param


def build_pull_param(user, str_id, mode="dev"):
    """
    str_id: 将作为key，用于异步查询结果
    mode: dev or online
    """
    param = {
        "serviceName": "chatgpt_response_query_dataview",
        "cacheInterval": "-1",
        "queryConditions": {
            "messageKey": f"{user}_{str_id}"
        }
    }
    param.update(get_key_pairs(mode))
    return param


class ChatGPT(Inference):
    def __init__(self, conf_path: str) -> None:
        super().__init__()
        self._conf = ConfigFactory.parse_file(conf_path)
        self.conf = None

        self.reset_conf()
        self.description()

    def description(self):
        logger.info(f"{'*'*100}")
        logger.info(f"models: {self.conf.req_template.queryConditions}")
        logger.info(f"{'*'*100}")
        return super().description()

    def __str__(self) -> str:
        return f"""{'*'*100}\nmodels: {self.conf.req_template.queryConditions}\n{'*'*100}"""

    def update_conf(self, new_param_conf: Dict):
        """
        自定义部分
        动态更新请求参数配置
        """
        if len(new_param_conf) == 0:
            return
        cur_param = deepcopy(self.conf.req_template)

        prefix_key = 'queryConditions'
        try:
            if cur_param.get('queryConditions.request_params') is not None:
                prefix_key = 'queryConditions.request_params'
        except:
            pass

        for k, v in new_param_conf.items():
            cur_param.put(f'$.{prefix_key}.{k}', v)
        
        self.conf.req_template = deepcopy(cur_param)
    
    def reset_conf(self):
        self.conf = deepcopy(self._conf)

    def gen_req(self, messages:List[dict], question_id=None):
        """生成请求"""
        # 阶段1：发送请求，请求带上工号和str_id，这些信息在查询时会用到
        param = deepcopy(self.conf.req_template)
        retry_times = 30
        headers = {'Content-Type': 'application/json'}
        try:
            if param.get('queryConditions.request_params') is not None:
                param.put('$.queryConditions.request_params.messages', messages)
            else:
                param.put('$.queryConditions.messages', messages)
        except:
            param.put('$.queryConditions.messages', messages)
        while retry_times > 0:
            if question_id is None:
                question_id = param.queryConditions.messageKey+f"_{time.time()}"
            param.put('$queryConditions.messageKey', question_id)
            post_data = {"encryptedParam": aes_encrypt(json.dumps(param))}
            # 异步请求，结果需要走另一个接口查询
            req_response = requests.post(self.conf.host, data=json.dumps(post_data), headers=headers).json()
            if self._check_respond(req_response):
                break
            else:
                pass
                # logger.warn(f'Retrying...')
            retry_times -= 1
        return req_response

    def _check_respond(self, content):
        if content['success'] is False:
            error_messages = html.unescape(content['data']['errorMessage'])
            # logger.warn(f'Failed to request!\nRequest response content: {error_messages}')
            return False
        return True

    def gen_pull(self, request):
        """获取答案"""
        # 阶段2：查询结果，根据工号和str_id查询。
        question_id = request['data']['values']['messageKey']
        param = deepcopy(self.conf.pull_template)
        param.put('$queryConditions.messageKey', question_id)
        headers = {'Content-Type': 'application/json'}
        retry_times = 50
        post_data = {"encryptedParam": aes_encrypt(json.dumps(param))}
        
        while retry_times > 0:    
            pull_response = requests.post(self.conf.host, data=json.dumps(post_data), headers=headers).json()
            flag = self._check_respond(pull_response)
            if flag and 'response' in pull_response['data']['values']:
                # logger.debug("Got pull response!")
                response = html.unescape(pull_response['data']['values']['response'])
                try:
                    delim = '"""'
                    if response.find('\'') < 0:
                        delim = '\''
                    elif response.find('"') < 0:
                        delim = '"'
                    elif response.find("'''") < 0:
                        delim = "'''"
                    return json.loads(ast.literal_eval(f"{delim}{response}{delim}"))
                except :
                    # logger.warn(f'Failed to parse response!')
                    return json.loads(response)
            time.sleep(3)
            retry_times -= 1
        raise RuntimeError('Fail to request!!!')

    def predict(self, messages: List[dict]):
        request = self.gen_req(messages)
        response = self.gen_pull(request)
        try:
            return response['choices'][0]['message']['content']
        except :
            print(response)


if __name__ == '__main__':
    __insert_answer__=""""足球诗人"这个称号常被用来形容那些在足球场上技术精湛、风格优雅，并且在比赛中展现出如诗般艺术美感的球员。这个称谓并非专指某一位球员，历史上多位球员因他们的球风而获得过这样的美誉。其中，巴西传奇球星苏格拉底（Sócrates）就经常被人们称为“足球场上的哲学家”或“足球诗人”，因其不仅球技出色，还拥有智慧深邃的形象。另外，阿根廷名将迭戈·马拉多纳（Diego Maradona）和西班牙中场大师安德烈斯·伊涅斯塔（Andrés Iniesta）等，也因其在场上的创造力和技术的诗意，被球迷和媒体赋予过类似的雅号。总的来说，“足球诗人”是一个泛指，用来赞美那些在足球运动中展现出非凡技艺与美感的运动员。"""
    __insert_prompt__="""足球诗人指的是谁"""
    predictor = ChatGPT("/mnt/nas_emx24/xuan.syx/code/code/AI_Portal/config/CLAUDE.conf")
    # predictor = ChatGPT("/mnt/nas_emx24/xuan.syx/code/code/AI_Portal/config/ChatGPT_alps.conf")
    messages = [
        {
            "role": "system",
            "content": "你作为一个聪明且严厉的assistant，能理解另外一个assistant [Bailing]的行为。你会对错误答案严厉的批评。\n<requirement>\n1. 记住，不要直接生成答案。\n2. 记住，不要生成改进后的示例。\n</requirement>"
        },
        {
            "role": "user",
            "content": f"<prompt>\n{__insert_prompt__}\n</prompt>\n我把<prompt> 给 [Bailing] 的答案是\n<answer>\n{__insert_answer__}\n</answer>\n如何他的答案不好，在满足<requirement>的前提下\n<task>\n1. 分析原始<prompt>的要求，分析[bailing]的<answer>是否满足要求。\n2. 帮我逐步分析下原因，并逐条给出改进的方向或者建议。\n</task>\n"
        }
    ]
    print(predictor.predict(messages))

