import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
up_dir = parent_dir
for i in range(3):
    sys.path.append(up_dir)
    up_dir = os.path.dirname(up_dir)
from kutils import DEBUG, INFO, WARN, ERROR
import utils as u

import json
from PIL import Image
from llm_service.longchen_requestor import request_longchen_gpt4o, request_longchen_gpt4o_pil
import torch
from qwenvl25.request import request_qwen
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor, Qwen2_5_VLProcessor
from transformers import Qwen2Tokenizer, Qwen2_5_VLPreTrainedModel
from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    NousFnCallPrompt,
    Message,
    ContentItem,
)
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize
from agent_function_call import ComputerUse
from qwen_vl_utils import process_vision_info

class ModelLoader():
    def __init__(self, model_path, gpu_num = None, force_bos_token = None) -> None:
        model_name = u.get_name(model_path, True)
        if 'UI-TARS' in model_name:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, 
                torch_dtype=torch.bfloat16, 
                attn_implementation="flash_attention_2",
                device_map="auto"
            )
        # self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.vl2_processor = AutoProcessor.from_pretrained(model_path)
        self.processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
        self.tokenizer = Qwen2Tokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.gpu_num = gpu_num
        self.max_new_tokens = 4096
        self.force_bos_token = force_bos_token

    def infer_vl2(self, messages):
        text = self.vl2_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # print(text)
        image_inputs, video_inputs = process_vision_info(messages)
        text = (lambda parts: parts[0])(text.rsplit('<|im_end|>\n<|im_start|>assistant', 1))
        inputs = self.vl2_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.vl2_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        return output_text[0]

    def infer_prompt(self, sys_prompt, prompt, img_files):
        messages = [
            {
                "role": "system", 
                "content": sys_prompt
            },
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text", 
                        "text": prompt
                    }
                ]
            }
        ]
        images = []
        for img_file in img_files:
            image = Image.open(img_file)
            image_local_path = "file://" + img_file
            messages[1]['content'].append({ "image": image_local_path })
            images.append(image)
        input_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[input_text], images=images, padding=True, return_tensors="pt")
        return self.__inference(inputs), messages

    def infer_messages(self, messages, img_files):
        images = []
        for img_file in img_files:
            image = Image.open(img_file)
            images.append(image)
        input_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[input_text], images=images, padding=True, return_tensors="pt")
        return self.__inference(inputs)

    def __inference(self, inputs):
        if self.gpu_num: inputs = inputs.to(f'cuda:{self.gpu_num}')
        else: inputs = inputs.to('cuda')
        if self.force_bos_token:
            think_token_id = self.tokenizer.convert_tokens_to_ids(self.force_bos_token)
            output_ids = self.model.generate(
                **inputs, 
                max_new_tokens=self.max_new_tokens, 
                forced_bos_token_id=think_token_id, 
                decoder_start_token_id=think_token_id
            )
        else:
            output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        return output_text[0]

    def inference_cua(self, screenshot_path, user_query):
        input_image = Image.open(screenshot_path)
        factor = self.processor.image_processor.patch_size * self.processor.image_processor.merge_size
        resized_height, resized_width = smart_resize(
            input_image.height,
            input_image.width,
            factor=factor,
            min_pixels=self.processor.image_processor.min_pixels,
            max_pixels=self.processor.image_processor.max_pixels,
        )
        
        # Initialize computer use function
        computer_use = ComputerUse(
            cfg={"display_width_px": resized_width, "display_height_px": resized_height}
        )

        # Build messages
        message = NousFnCallPrompt().preprocess_fncall_messages(
            messages=[
                Message(role="system", content=[ContentItem(text="You are a helpful assistant.")]),
                Message(role="user", content=[
                    ContentItem(text=user_query),
                    ContentItem(image=f"file://{screenshot_path}")
                ]),
            ],
            functions=[computer_use.function],
            lang=None,
        )
        message = [msg.model_dump() for msg in message]

        # Process input
        text = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[input_image], padding=True, return_tensors="pt").to('cuda')

        # Generate output
        output_ids = self.model.generate(**inputs, max_new_tokens=4096)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)[0]

        return output_text, message

class SimpleModel():
    def __init__(self) -> None:
        self.model_name = 'gpt-4o'

    def __request_vlm(self, sys_prompt, query_prompt, img_file):
        response = request_longchen_gpt4o(sys_prompt, query_prompt, img_file, self.model_name)
        return response

    def infer_prompt(self, sys_prompt, prompt, img_files):
        return self.infer(sys_prompt, prompt, img_files), None

    def infer_pil(self, sys_prompt, prompt, pil_img):
        return request_longchen_gpt4o_pil(sys_prompt, prompt, pil_img, self.model_name)
    
    def infer_vl2(self, messages):
        raise ValueError('Not implement')
                    
    def infer_messages(self, messages, img_files):
        raise ValueError('Not implement')

    def infer(self, sys_prompt, query_prompt, img_file):
        txt_result = self.__request_vlm(sys_prompt, query_prompt, img_file[0])
        return txt_result

class ModelCache():
    def __init__(self, cache_path, model_name) -> None:
        self.cache_path = cache_path
        self.start_time = u.get_time()
        self.model_name = model_name
        u.mkdir(self.cache_path)
        year_month = self.start_time[:6]
        day = self.start_time[6:8]
        hour = self.start_time[8:10]
        ms = self.start_time[10:]
        self.output_path = self.cache_path + '/' + year_month + '/' + day + '/' + hour + '/' + ms + '/'
        u.mkdir(self.output_path)
        self.cnt = 0
        self.split_str = 100*'-'

    def __request_vlm(self, sys_prompt, query_prompt, img_file):
        if self.model_name == 'gpt-4o':
            response = request_longchen_gpt4o(sys_prompt, query_prompt, img_file, self.model_name)
            return response
        elif self.model_name == 'qwen':
            response = request_qwen(sys_prompt, query_prompt, img_file)
            return response
        else:
            raise ValueError('error model name')

    def request(self, sys_prompt, query_prompt, img_file):
        prompt_hash = u.gen_fixed_length_hash(self.model_name + sys_prompt + query_prompt, 16)
        if img_file:
            img_file_hash = u.gen_fixed_length_hash(img_file, 16)
            cache_hash = prompt_hash + img_file_hash
        else:
            cache_hash = prompt_hash
        cache_files = u.get_all_filenames(self.cache_path)
        matched_strings = u.get_string_in_list(cache_files, cache_hash)
        filename = f'{u.get_time()[-6:]}_{self.cnt}_{cache_hash}.txt'
        filename = self.output_path + '/' + filename
        if len(matched_strings) > 0:
            cache_data = u.read_txt(matched_strings[0])
            txt_result= u.extract_text(cache_data, 'answer:\n', None)
            # if txt_result != '': u.write_txt(filename, cache_data)
        else:
            txt_result = self.__request_vlm(sys_prompt, query_prompt, img_file)
            if txt_result: u.mkdir(self.output_path)
            u.write_txt(filename, 'prompt:\n' + query_prompt + f'\n{self.split_str}\nanswer:\n' + txt_result)
        self.cnt += 1
        return txt_result
