import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
up_dir = parent_dir
for i in range(3):
    sys.path.append(up_dir)
    up_dir = os.path.dirname(up_dir)
from kutils import DEBUG, INFO, WARN, ERROR
import utils as u
from openai import OpenAI
import os
from PIL import Image

class OpenRouter():
    def __init__(self, config):
        self.model = config['model']
        self.temperature = config['temperature']
        self.max_tokens = config['max_tokens']
        self.client = OpenAI(
            api_key = config['api_key'],
            base_url = config['base_url'],
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

    def infer_messages(self, messages):
        response = self.client.chat.completions.create(
            model = self.model,
            messages = messages,
            temperature = self.temperature,
            max_tokens = self.max_tokens,
        )
        response = response.choices[0].message.content
        return response

if __name__ == "__main__":
    API_KEY = os.getenv("OPENROUTER_KEY")
    kq_config = {
        'api_key': API_KEY,
        'model': 'openai/gpt-5-chat',
        'base_url': "https://openrouter.ai/api/v1",
        'temperature': 0.0,
        'max_tokens': 4096,
    }

    kq = OpenRouter(kq_config)
    prompt = ""
    image_file = f'{u.get_git()}/0_saved_methods/qwenvl25/cookbooks/assets/universal_recognition/unireco_bird_example.jpg'
    img = Image.open(image_file)
    response = kq.infer('', prompt, img)
    DEBUG(response)