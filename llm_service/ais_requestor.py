import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
up_dir = parent_dir
for i in range(3):
    sys.path.append(up_dir)
    up_dir = os.path.dirname(up_dir)
from kutils import DEBUG, INFO, WARN, ERROR
import utils as u
from tqdm import tqdm
from PIL import Image
import pandas as pd
import json
import io
import requests
from openai import OpenAI

class KevinAISRequestor():
    def __init__(self, config):
        self.model = config['model']
        self.temperature = config['temperature']
        self.max_tokens = config['max_tokens']
        self.client = OpenAI(
            api_key = config['api_key'],
            base_url = config['base_url'],
            timeout = 600000,
        )

    def infer(self, system, prompt, pil_img):
        messages = \
        [
            {
                'role': 'system', 
                'content': system
            }, 
            {
                'role': 'user', 
                'content': [
                    {
                        'type': 'text', 
                        'text': prompt
                    }, 
                    {
                        'type': 'image_url', 
                        'image_url': { 
                            'url': 'data:image/png;base64,' 
                        }
                    }
                ]
            }
        ]

        image_base64 = u.pil_image_to_base64(pil_img)
        messages[1]['content'][1]['image_url']['url'] += image_base64
        return self.infer_messages(messages)

    def infer_with_input_img(self, system, prompt, input_img, screenshot):
        messages = \
        [
            {
                'role': 'system', 
                'content': system
            }, 
            {
                'role': 'user', 
                'content': [
                    {
                        'type': 'text', 
                        'text': prompt
                    }, 
                    {
                        "type": "text",
                        "text": "IMAGES: (0) user input image"
                    },
                    {
                        'type': 'image_url', 
                        'image_url': { 
                            'url': 'data:image/png;base64,' 
                        }
                    },
                    {
                        "type": "text",
                        "text": "(1) current page screenshot"
                    },
                    {
                        'type': 'image_url', 
                        'image_url': { 
                            'url': 'data:image/png;base64,' 
                        }
                    }
                ]
            }
        ]
        input_image_base64 = u.pil_image_to_base64(input_img)
        messages[1]['content'][2]['image_url']['url'] += input_image_base64
        ss_image_base64 = u.pil_image_to_base64(screenshot)
        messages[1]['content'][4]['image_url']['url'] += ss_image_base64
        return self.infer_messages(messages)

    def infer_messages(self, messages) -> str:
        response = self.client.chat.completions.create(
            model = self.model,
            messages = messages,
            temperature = self.temperature,
            max_tokens = self.max_tokens,
        )
        response = response.choices[0].message.content
        return response

def check_providers(providers):
    kq_config = {
        'api_key': '123',
        'model': '',
        'base_url': "https://agi-pre.alipay.com/api",
        'temperature': 0.0,
        'max_tokens': 4096,
    }
    for provider in providers:
        kq_config['model'] = provider
        kq = KevinAISRequestor(kq_config)
        prompt = ""
        image_file = f'{u.get_git()}/asset/unireco_bird_example.jpg'
        img = Image.open(image_file)
        try:
            response = kq.infer('', prompt, img)
            if isinstance(response, str): 
                INFO(f'provider {provider} good')
        except Exception as e:
            DEBUG(f'provider {provider} {e}')
            return False
    return True

if __name__ == "__main__":

    providers = [
        'KevinQwen', 
        'KevinQwen2', 
        'KevinQwen3', 
        'KevinQwen4', 
        'zg-qw72b-h1', 
        'zg-qw72b-h2',
        'KevinBlip2'
    ]

    check_providers(providers)