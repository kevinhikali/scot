import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
up_dir = parent_dir
for i in range(3):
    sys.path.append(up_dir)
    up_dir = os.path.dirname(up_dir)
from kutils import DEBUG, INFO, WARN, ERROR
import utils as u
import argparse
import json
from typing import Any, Optional

import tiktoken
from beartype import beartype
from PIL import Image

from agent.prompts import *
from browser_env import Trajectory
from browser_env.actions import (
    Action,
    ActionParsingError,
    create_id_based_action,
    create_none_action,
    create_playwright_action,
    create_vision_action,
)
from browser_env.utils import Observation, StateInfo
from llms import (
    # call_llm,
    # generate_from_huggingface_completion,
    # generate_from_openai_chat_completion,
    # generate_from_openai_completion,
    lm_config,
)
from llms.tokenizers import Tokenizer
from visualwebarena.agent.prompts.prompt_constructor import MultimodalCoTPromptConstructor, KevinMultimodalCoTPromptConstructor, KevinSoMPromptConstructor

from llm_service.longchen_requestor import request_messages
from llm_service.ais_requestor import KevinAISRequestor
from dataset.multimodel_mind2web.prompt_parser import sys_prompt
from dataset.multimodel_mind2web.prompt_parser import parse_response_vwa as parse_response
from dataset.multimodel_mind2web.prompt_parser import prompt_bbox_vwa as prompt_f
import mas_prompts as mp

from dataset.visualwebarena.agent.tool_agents import SimJudger

class Agent:
    """Base class for the agent"""

    def __init__(self, *args: Any) -> None:
        pass

    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: Any
    ) -> Action:
        """Predict the next action given the observation"""
        raise NotImplementedError

    def reset(
        self,
        test_config_file: str,
    ) -> None:
        raise NotImplementedError


class TeacherForcingAgent(Agent):
    """Agent that follows a pre-defined action sequence"""

    def __init__(self) -> None:
        super().__init__()

    def set_action_set_tag(self, tag: str) -> None:
        self.action_set_tag = tag

    def set_actions(self, action_seq: str | list[str]) -> None:
        if isinstance(action_seq, str):
            action_strs = action_seq.strip().split("\n")
        else:
            action_strs = action_seq
        action_strs = [a.strip() for a in action_strs]

        actions = []
        for a_str in action_strs:
            try:
                if self.action_set_tag == "playwright":
                    cur_action = create_playwright_action(a_str)
                elif self.action_set_tag == "id_accessibility_tree":
                    cur_action = create_id_based_action(a_str)
                else:
                    raise ValueError(
                        f"Unknown action type {self.action_set_tag}"
                    )
            except ActionParsingError as e:
                cur_action = create_none_action()

            cur_action["raw_prediction"] = a_str
            actions.append(cur_action)

        self.actions: list[Action] = actions

    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: Any
    ) -> Action:
        """Predict the next action given the observation"""
        return self.actions.pop(0)

    def reset(
        self,
        test_config_file: str,
    ) -> None:
        with open(test_config_file) as f:
            ref_actions = json.load(f)["reference_action_sequence"]
            tag = ref_actions["action_set_tag"]
            action_seq = ref_actions["action_sequence"]
            self.set_action_set_tag(tag)
            self.set_actions(action_seq)


class PromptAgent(Agent):
    """prompt-based agent that emits action given the history"""

    @beartype
    def __init__(
        self,
        action_set_tag: str,
        lm_config: lm_config.LMConfig,
        prompt_constructor: PromptConstructor | MultimodalCoTPromptConstructor | KevinMultimodalCoTPromptConstructor | KevinSoMPromptConstructor,
        captioning_fn = None,
    ) -> None:
        super().__init__()
        self.lm_config = lm_config
        self.prompt_constructor = prompt_constructor
        self.action_set_tag = action_set_tag
        self.captioning_fn = captioning_fn
        self.multimodal_inputs = True

        kq_config = {
            'api_key': '123',
            'model': lm_config.provider,
            'base_url': "https://agi-pre.alipay.com/api",
            'temperature': 0.0,
            'max_tokens': 4096,
        }
        self.kq = KevinAISRequestor(kq_config)

    def set_action_set_tag(self, tag: str) -> None:
        self.action_set_tag = tag

    @beartype
    def next_action(
        self, 
        trajectory: Trajectory, 
        intent: str, 
        meta_data: dict[str, Any], 
        images: Optional[list[Image.Image]] = None,
        output_response: bool = False
    ) -> Action:
        # Create page screenshot image for multimodal models.
        if self.multimodal_inputs:
            page_screenshot_arr = trajectory[-1]["observation"]["image"]
            page_screenshot_img = Image.fromarray(
                page_screenshot_arr
            )  # size = (viewport_width, viewport_width)

        # Caption the input image, if provided.
        if images is not None and len(images) > 0:
            if self.captioning_fn is not None:
                image_input_caption = ""
                for image_i, image in enumerate(images):
                    if image_i == 0:
                        image_input_caption += f'Input image {image_i+1}: "{self.captioning_fn([image])[0]}"'
                    else:
                        image_input_caption += f'input image {image_i+1}: "{self.captioning_fn([image])[0]}"'
                    if len(images) > 1:
                        image_input_caption += ", "
                # Update intent to include captions of input images.
                intent = f"{image_input_caption}\nIntent: {intent}"
            elif not self.multimodal_inputs:
                print(
                    "WARNING: Input image provided but no image captioner available."
                )

        if self.multimodal_inputs:
            messages = self.prompt_constructor.construct(
                trajectory, intent, page_screenshot_img, images, meta_data
            )
        else:
            messages = self.prompt_constructor.construct(
                trajectory, intent, meta_data
            )
        lm_config = self.lm_config
        n = 0
        while True:
            # response = call_llm(lm_config, prompt)
            # u.write_json(f'{u.get_time()}.json', messages)

            try:
                model = lm_config.model
                if 'qwen' in model:
                    response = self.kq.infer_messages(messages)
                elif 'gpt' in model:
                    response = request_messages(messages)
                else:
                    raise ValueError(lm_config)
            except Exception as e:
                ERROR(e)
                response = 'stop []'

            force_prefix = self.prompt_constructor.instruction[
                "meta_data"
            ].get("force_prefix", "")
            response = f"{force_prefix}{response}"
            if output_response:
                print(f'Agent: {response}', flush=True)
            n += 1
            try:
                parsed_response = self.prompt_constructor.extract_action(
                    response
                )
                if self.action_set_tag == "id_accessibility_tree":
                    action = create_id_based_action(parsed_response)
                elif self.action_set_tag == "playwright":
                    action = create_playwright_action(parsed_response)
                elif self.action_set_tag == "som":
                    action = create_id_based_action(parsed_response)
                else:
                    raise ValueError(
                        f"Unknown action type {self.action_set_tag}"
                    )
                action["raw_prediction"] = response
                break
            except ActionParsingError as e:
                if n >= lm_config.gen_config["max_retry"]:
                    action = create_none_action()
                    action["raw_prediction"] = response
                    break

        return action

    def reset(self, test_config_file: str) -> None:
        pass

class VisionAgent(Agent):
    """prompt-based agent that emits action given the history"""

    @beartype
    def __init__(
        self,
        action_set_tag: str,
        lm_config: lm_config.LMConfig,
        prompt_constructor: PromptConstructor | MultimodalCoTPromptConstructor | KevinMultimodalCoTPromptConstructor,
        captioning_fn = None,
    ) -> None:
        super().__init__()
        self.lm_config = lm_config
        self.prompt_constructor = prompt_constructor
        self.action_set_tag = action_set_tag
        self.captioning_fn = captioning_fn
        self.multimodal_inputs = True
        kq_config = {
            'api_key': '123',
            'model': lm_config.provider,
            'base_url': "https://agi-pre.alipay.com/api",
            'temperature': 0.0,
            'max_tokens': 4096,
        }
        self.kq = KevinAISRequestor(kq_config)

    def set_action_set_tag(self, tag: str) -> None:
        self.action_set_tag = tag

    def map_to_vwa_actions(self, action_info):
        pred_action_history = action_info['pred_action_history']
        pred_action_description = action_info['pred_action_description']
        pred_action = action_info['pred_action']
        pred_action_type = action_info['pred_action_type']
        pred_bbox = action_info['pred_bbox']
        pred_type_value = action_info['pred_type_value']
        pred_click_point = action_info['pred_click_point']
        parse_error_msg = action_info['parse_error_msg']
        return pred_action

    def filter_bboxes(self, bboxes, w, h):
        to_be_del = []
        for idx, box in bboxes.items():
            if box['x'] < 0 or box['x'] > w or box['y'] < 0 or box['y'] > h:
                to_be_del.append(idx)
        for idx in to_be_del:
            del bboxes[idx]
        return bboxes

    @beartype
    def next_action(
        self, 
        trajectory: Trajectory, 
        intent: str, 
        meta_data: dict[str, Any], 
        images: Optional[list[Image.Image]] = None,
        output_response: bool = False
    ) -> Action:
        # Create page screenshot image for multimodal models.
        if self.multimodal_inputs:
            page_screenshot_arr = trajectory[-1]["observation"]["ori_image"]
            page_screenshot_img = Image.fromarray(page_screenshot_arr)  # size = (viewport_width, viewport_width)

        bboxes = meta_data['bbox']

        n = 0
        while True:
            # task, action history, memo, tabs, hint
            prompt = prompt_f.format(intent, meta_data["action_history"], '', '', meta_data['hint'])
            if images:
                input_img = images[0]
                response = self.kq.infer_with_input_img(
                    sys_prompt, 
                    prompt,
                    input_img, 
                    page_screenshot_img)
            else:
                response = self.kq.infer(
                    sys_prompt,
                    prompt,
                    page_screenshot_img)

            action_info = parse_response(response, 'bbox')
            force_prefix = self.prompt_constructor.instruction["meta_data"].get("force_prefix", "")
            response = f"{force_prefix}{response}"
            if output_response: print(f'{response}', flush=True)

            n += 1
            try:
                action = create_vision_action(action_info, bboxes)
                action["raw_prediction"] = response
                action['action_info'] = action_info
                break
            except ActionParsingError as e:
                INFO(e)
                if n >= self.lm_config.gen_config["max_retry"]:
                    action = create_none_action()
                    action["raw_prediction"] = response
                    break

        action['prompt'] = prompt
        action['action_info'] = action_info
        return action

    def reset(self, test_config_file: str) -> None:
        pass

class MultiAgent(Agent):
    @beartype
    def __init__(
        self,
        action_set_tag: str,
        lm_config: lm_config.LMConfig,
        prompt_constructor: PromptConstructor | 
        MultimodalCoTPromptConstructor | 
        KevinMultimodalCoTPromptConstructor | 
        KevinSoMPromptConstructor,
        captioning_fn = None,
    ) -> None:
        super().__init__()
        self.lm_config = lm_config
        self.prompt_constructor = prompt_constructor
        self.action_set_tag = action_set_tag
        self.captioning_fn = captioning_fn
        self.multimodal_inputs = True

        if 'qwen' in self.lm_config.model:
            kq_config = {
                'api_key': '123',
                'model': lm_config.provider,
                'base_url': "https://agi-pre.alipay.com/api",
                'temperature': 0.0,
                'max_tokens': 4096,
            }
            self.kq = KevinAISRequestor(kq_config)

    def set_action_set_tag(self, tag: str) -> None:
        self.action_set_tag = tag

    def get_action_agent_messages(
            self, 
            system_prompt, 
            task, 
            url, 
            obs,
            tabs,
            action_history,
            action_hint,
            input_img, 
            som_page_screenshot_img,
        ):
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt,
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        'type': 'text', 
                        'text': f'Task: {task}', 
                    },
                    {
                        'type': 'text', 
                        'text': f'Current web page\'s URL: {url}',
                    
                    },
                    {
                        "type": "text",
                        "text": f'Observations: {obs}',
                    },
                    {
                        "type": "text",
                        "text": f'Open tabs: {tabs}',
                    },
                    {
                        "type": "text",
                        "text": f'Action history: {action_history}',
                    },
                    {
                        "type": "text",
                        "text": f'Action hint by human adviser: {action_hint}',
                    },
                ]
            }
        ]

        if input_img:
            input_img_msg = \
            [
                {
                    "type": "text",
                    "text": "User input image:"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        'url': 'data:image/png;base64,' 
                    }
                }
            ]
            input_image_base64 = u.pil_image_to_base64(input_img)
            input_img_msg[-1]['image_url']['url'] += input_image_base64
            messages[1]['content'] += input_img_msg
        
        page_img_msg = \
        [
            {
                "type": "text",
                "text": "Current page screenshot with interactable bounding boxes:"
            },
            {
                "type": "image_url",
                "image_url": {
                    'url': 'data:image/png;base64,' 
                }
            }
        ]
        ss_image_base64 = u.pil_image_to_base64(som_page_screenshot_img)
        page_img_msg[-1]['image_url']['url'] += ss_image_base64
        messages[1]['content'] += page_img_msg 
        return messages

    def __request(self, prompt, input_img, som_page_screenshot_img):
        if input_img:
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a helpful assistant." 
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            'type': 'text', 
                            'text': prompt, 
                        }, 
                        {
                            "type": "text",
                            "text": "IMAGES: (1) user input image"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                'url': 'data:image/png;base64,' 
                            }
                        },
                        {
                            "type": "text",
                            "text": "(2) current page screenshot with interactable bounding boxes"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                'url': 'data:image/png;base64,' 
                            }
                        }
                    ]
                }
            ]
            input_image_base64 = u.pil_image_to_base64(input_img)
            messages[1]['content'][2]['image_url']['url'] += input_image_base64
            ss_image_base64 = u.pil_image_to_base64(som_page_screenshot_img)
            messages[1]['content'][4]['image_url']['url'] += ss_image_base64
        else:
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a helpful assistant." 
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            'type': 'text', 
                            'text': prompt, 
                        }, 
                        {
                            "type": "text",
                            "text": "IMAGES: (1) current page screenshot"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                'url': 'data:image/png;base64,' 
                            }
                        }
                    ]
                }
            ]
            ss_image_base64 = u.pil_image_to_base64(som_page_screenshot_img)
            messages[1]['content'][2]['image_url']['url'] += ss_image_base64

        while 1:
            try:
                response = self.kq.infer_messages(messages)
                break
            except Exception as e:
                ERROR(f'{self.lm_config} {e}')
                response = 'stop []'
            u.wait(1)
        
        return response

    def __parse_master_response(self, response):
        name = ''
        subtask = ''
        error_msg = ''
        response_results = response
        try: response_results = u.extract_text(response, '```json', '```')[0]
        except Exception as e: error_msg += f'{response}, error at extract text {e}\n'
        # try: next_member = u.extract_text(response_results, '\"name\": \"', '\",')[0]
        # except Exception as e: error_msg += f'{response}, error at name {e}\n'
        try: name = u.extract_text(response_results, '\"name\": \"', '\"\n}')[0]
        except Exception as e: 
            error_msg += f'{response}, error at name {e}\n'
            try: name = u.extract_text(response_results, '\'name\': \'', '\'\n}')[0]
            except Exception as e: error_msg += f'{response}, error at name {e}\n'
        return name, error_msg

    def __parse_shopping_guide_response(self, response):
        reason = ''
        description = ''
        category = ''
        error_msg = ''
        response_results = response
        try: response_results = u.extract_text(response, '```json', '```')[0]
        except Exception as e: error_msg += f'{response}, error at json {e}\n'
        try: reason = u.extract_text(response_results, '\"reason\": \"', '\",')[0]
        except Exception as e: error_msg += f'{response}, error at reason {e}\n'
        try: description = u.extract_text(response_results, '\"description\": \"', '\",')[0]
        except Exception as e: error_msg += f'{response}, error at description {e}\n'
        try: category = u.extract_text(response_results, '\"category\": \"', '\"\n}')[0]
        except Exception as e: error_msg += f'{response}, error at category {e}\n'
        return reason, description, category

    def __parse_action_agent_response(self, response):
        description = ''
        action = ''
        error_msg = ''
        response_results = response
        try: response_results = u.extract_text(response_results, '```json', '```')[0]
        except Exception as e: error_msg += f'{response}, error at json {e}\n'

        try: 
            description = u.extract_text(response_results, '\"action_description\": \"', '\",')[0]
        except Exception as e: 
            try:
                description = u.extract_text(response_results, '\'action_description\': \'', '\',')[0]
            except Exception as e:
                error_msg += f'{response}, error at description {e}\n'

        try: 
            action = u.extract_text(response_results, '\"action\": \"', '\"\n}')[0]
        except Exception as e:
            try:
                action = u.extract_text(response_results, '\'action\': \'', '\'\n}')[0]
            except Exception as e: 
                error_msg += f'{response}, error at action {e}\n'

        action = f'```{action}```'
        return description, action, error_msg

    @beartype
    def next_action(
        self, 
        trajectory: Trajectory, 
        intent: str, 
        meta_data: dict[str, Any], 
        images: Optional[list[Image.Image]] = None,
        output_response: bool = False
    ) -> Action:
        # Create page screenshot image for multimodal models.
        ori_page_screenshot_arr = trajectory[-1]["observation"]["ori_image"]
        ori_page_screenshot_img = Image.fromarray(ori_page_screenshot_arr)
        som_page_screenshot_arr = trajectory[-1]["observation"]["image"]
        som_page_screenshot_img = Image.fromarray(som_page_screenshot_arr)

        # Caption the input image, if provided.
        image_input_caption = ''
        input_img = None
        if images is not None and len(images) > 0:
            if self.captioning_fn is not None:
                for image_i, image in enumerate(images):
                    if image_i == 0:
                        image_input_caption += f'Input image {image_i+1}: "{self.captioning_fn([image])[0]}"'
                    else:
                        image_input_caption += f'input image {image_i+1}: "{self.captioning_fn([image])[0]}"'
                    if len(images) > 1:
                        image_input_caption += ", "
                # Update intent to include captions of input images.
                intent = f"{image_input_caption}\nTask: {intent}"
            elif not self.multimodal_inputs:
                print("WARNING: Input image provided but no image captioner available.")
            input_img = images[0]
        
        page_text = trajectory[-1]['observation']['text']
        page_texts = page_text.split('\n')
        page_texts = [a.replace('  ', '') for a in page_texts if not a.startswith('[]')]
        page_text = '\n'.join(page_texts)

        n = 0
        while True:
            action_info = {
                'pred_action_history': '',
                'pred_action_description': '',
                'pred_action': '',
                'pred_action_type': '',
                'pred_bbox': '',
                'pred_type_value': '',
                'pred_click_point': '',
                'parse_error_msg': '',
                'content_to_memo': '',
            }
            action_history = meta_data["action_history"]
            pr_master = mp.master.format(
                TASK = intent, 
                ACTION_HISTORY = action_history
            )

            master_response = self.__request(pr_master, input_img, som_page_screenshot_img)

            slave_response = 'In summary, the next action I will perform is ```stop []```'
            if 'image_searcher' in master_response:
                sim_judger=SimJudger(LLM_MODEL_NAME="zg-qw72b-h4", LLM_API_KEY="xx", LLM_BASE_URL="https://agi.alipay.com/api")
                max_sims,max_item= sim_judger.get_item(input_img,meta_data["page"],sim_method="ahash_similarity")
                if max_item:
                    comments_url = max_item["comments_url"]
                    if comments_url:
                        slave_response = f'In summary, the next action I will perform is ```goto [http://localhost:9999{comments_url}]```'

            elif 'shopping_guide' in master_response:
                cate_file = f'{u.get_nas()}/gui_dataset/visualwebarena/shopping_categories.json'
                raw_categories = u.read_json(cate_file)
                categories = {}
                for raw_cate in raw_categories:
                    categories[raw_cate['class_name']] = raw_cate['class_url']
                pr_shopping_guide = mp.shopping_guide.format(
                    TASK = intent,
                    CATEGORIES = json.dumps(list(categories.keys()))
                )
                slave_response = self.__request(
                    pr_shopping_guide, 
                    input_img, 
                    ori_page_screenshot_img
                )
                slave_reason, slave_desc, slave_response = self.__parse_shopping_guide_response(slave_response)
                action_info['pred_action_description'] = f'shopping_guide: {slave_reason}\n{slave_desc}'
                category_url = categories.get(slave_response, '')
                category_url += 'product_list_order=price'
                slave_response = \
                    f'In summary, the next action I will perform is ```goto [{category_url}]```'

            action_hint = slave_response
            state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]
            obs = state_info["observation"]['text']
            page = state_info["info"]["page"]
            url = page.url
            tabs = ['current page', 'wiki']
            aa_messages = self.get_action_agent_messages(
                mp.action_agent, intent, url, obs, tabs, 
                action_history, action_hint, 
                input_img, som_page_screenshot_img)
            response = self.kq.infer_messages(aa_messages)
            aa_desc, response, aa_error_msg = self.__parse_action_agent_response(response)
            action_info['pred_action_description'] += f'\naction_agent: {aa_desc}'

            force_prefix = self.prompt_constructor.instruction["meta_data"].get("force_prefix", "")
            response = f"{force_prefix}{response}"
            if output_response: INFO(f'Agent: {response}')
            n += 1
            try:
                parsed_response = \
                    self.prompt_constructor.extract_action(response)
                if self.action_set_tag == "id_accessibility_tree":
                    action = create_id_based_action(parsed_response)
                elif self.action_set_tag == "playwright":
                    action = create_playwright_action(parsed_response)
                elif self.action_set_tag == "som":
                    action = create_id_based_action(parsed_response)
                else:
                    raise ValueError(f"Unknown action type {self.action_set_tag}")
                action["raw_prediction"] = response
                break
            except ActionParsingError as e:
                if n >= self.lm_config.gen_config["max_retry"]:
                    action = create_none_action()
                    action["raw_prediction"] = response
                    break

        action['action_info'] = action_info
        return action

    def reset(self, test_config_file: str) -> None:
        pass

def construct_agent(args: argparse.Namespace, captioning_fn=None) -> Agent:
    llm_config = lm_config.construct_llm_config(args)

    default_provider = 'openai'
    default_model = 'gpt-3.5-turbo-1106'
    tokenizer = Tokenizer(default_provider, default_model)

    agent: Agent
    if args.mode == "som":
        prompt_constructor = MultimodalCoTPromptConstructor(
            args.instruction_path, lm_config=llm_config, tokenizer=tokenizer
        )
        agent = PromptAgent(
            action_set_tag=args.action_set_tag,
            lm_config=llm_config,
            prompt_constructor=prompt_constructor,
            captioning_fn=captioning_fn
        )
    elif args.mode == "vision":
        prompt_constructor = KevinMultimodalCoTPromptConstructor(
            args.instruction_path, lm_config=llm_config, tokenizer=tokenizer
        )
        agent = VisionAgent(
            action_set_tag=args.action_set_tag,
            lm_config=llm_config,
            prompt_constructor=prompt_constructor,
            captioning_fn=captioning_fn
        )
    elif args.mode == "mas":
        prompt_constructor = KevinSoMPromptConstructor(
            args.instruction_path, lm_config=llm_config, tokenizer=tokenizer
        )
        agent = MultiAgent(
            action_set_tag=args.action_set_tag,
            lm_config=llm_config,
            prompt_constructor=prompt_constructor,
            captioning_fn=captioning_fn
        )
    else:
        raise NotImplementedError(
            f"agent type {args.mode} not implemented"
        )
    return agent
