import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
up_dir = parent_dir
for i in range(3):
    sys.path.append(up_dir)
    up_dir = os.path.dirname(up_dir)
from kutils import DEBUG, INFO, WARN, ERROR
import utils as u
from datasets import load_dataset
from tqdm import tqdm
import pyarrow.json as pajson
from PIL import Image
import pandas as pd
import json
from dataset.multimodel_mind2web.infer import map_action_desc_history
from llm_service.model_cache import ModelCache, ModelLoader, SimpleModel
from llm_service.vllm_inferer import VLLMInferer
import copy
from dataset_utils import expand_box
from infer_utils import map_to_actions
from prompt_parser import prompt_bbox_250714 as user_prompt_f
from prompt_parser import sys_prompt

prompt_f = \
'''You will be provided with a task, a chosen action and a screenshot of a webpage with a red bounding box indicating the interacting area of chosen action in the page.
Your job is to set chosen action as "Candidate action number 1" and infer 2 more candidate actions that you think can also be executed in this step to accomplish the task and give me the intentions and consequences of the all 3 actions.
Finally, since we have the chosen action, give me a reason why we choose it combining information you observed in the screenshot and actions has been executed in the action history.

Your task:
{task}

The action history:
{action_history}

Output in the following json format:
```json
{{
    "Candidate action number 1": {{
        "intention": "",
        "consequence": ""
    }},
    "Candidate action number 2": {{
        "intention": "",
        "consequence": ""
    }},
    "Candidate action number 3": {{
        "intention": "",
        "consequence": ""
    }},
    "reason": ""
}}
```'''

cot = \
'''<think>Okay, my task is {query_str}.
First, I should work out the candidate actions according to action history and the page screenshot. There are {n_candidate_action} actions I can take.
{candidate_action_cot}{select_reason}
Ok, my intention is identified, I need to provide the specific step for execution:
{gt_action_str}
Finally, I should give the final formatted result:</think>'''

gt_answer = \
'''{cot}<answer>```json
{{
    "action_description": "{action_description}",
    "action": "{action}"
}}
```</answer>'''

class SCOTInfer():
    def __init__(self, config):
        INFO('n_chunks chunk_index inferer(vllm, transformers, gpt-4o)')
        cfg = config
        n_chunks = int(cfg.argv[1])
        chunk_index = int(cfg.argv[2])
        inferer = cfg.argv[3]

        if u.get_os() == 'mac': NAS_PATH = f'{u.get_home()}/data/'
        else: NAS_PATH = '/mnt/agent-s1/common/public/kevin/'

        dataset_name = 'MultiModel_Mind2Web'
        dataset_path = f'{NAS_PATH}/gui_dataset/{dataset_name}/'
        sample_path = f'{dataset_path}/sample/'
        output_path = f'{NAS_PATH}/gui_dataset/{dataset_name}/train/anno_250618_{inferer}/'
        u.mkdir(output_path)
        
        raw_anno_file = f'{sample_path}/cross_task/block_sample.jsonl'
        self.anno_file = f'{output_path}/val_{n_chunks}_{chunk_index}.json'
        self.img_path = f'{sample_path}/cross_task/image_blocks/'
        self.gt_img_path = f'{dataset_path}/gt_imgs/cross_task/'

        # raw_anno_file = f'{sample_path}/train/block_sample.jsonl'
        # self.anno_file = f'{output_path}/train_{n_chunks}_{chunk_index}.json'
        # self.img_path = f'{sample_path}/train/image_blocks/'
        # self.gt_img_path = f'{dataset_path}/gt_imgs/train/'
        u.mkdir(self.gt_img_path)

        if inferer == 'vllm':
            model_path = f'{NAS_PATH}/model/Qwen2.5-VL-72B-Instruct/'
            self.mc = VLLMInferer('qwen2_5_vl', model_path)
        elif inferer == 'transformers':
            model_path = f'{NAS_PATH}/model/Qwen2.5-VL-72B-Instruct/'
            self.mc = ModelLoader(model_path)
        elif inferer == 'gpt-4o':
            self.mc = SimpleModel()
        else:
            raise ValueError('Error inferer ', inferer)

        raw_annos = u.read_json(raw_anno_file)

        self.annos_dict = {}
        for i in range(len(raw_annos)):
            step_content = raw_annos[i]
            anno_id = step_content['annotation_id']
            curr_step = step_content['step']
            if anno_id not in self.annos_dict.keys(): self.annos_dict[anno_id] = {}
            self.annos_dict[anno_id][curr_step] = step_content
            self.annos_dict[anno_id] = {key: self.annos_dict[anno_id][key] for key in sorted(self.annos_dict[anno_id].keys())}

        self.anno_keys = list(self.annos_dict.keys())
        n_all = len(self.anno_keys)
        n_sep = int(n_all / n_chunks)
        if chunk_index == (n_chunks - 1): self.anno_keys = self.anno_keys[chunk_index * n_sep:]
        else: self.anno_keys = self.anno_keys[chunk_index * n_sep: (chunk_index + 1) * n_sep]

    def infer(self):
        sg_anno = []
        for anno_id in tqdm(self.anno_keys):
            img_sw = []
            prebuild_action_history = []
            for curr_step in tqdm(self.annos_dict[anno_id].keys(), anno_id):
                step_content = self.annos_dict[anno_id][curr_step]
                anno_id = step_content['annotation_id']
                action_uid = step_content['action_uid']
                block_path = step_content['blocks_path']
                task = step_content['task']
                gt_action_type = step_content['operation']
                ori_operation = step_content['ori_operation']
                gt_action_value = step_content['value']
                target_blocks = step_content['target_blocks']
                ori_bbox = step_content['bbox']
                block_id = list(target_blocks.keys())[0]
                gt_action_desc_history = step_content['previous_actions']
                gt_action_desc_history = map_action_desc_history(gt_action_desc_history)
                n_step = step_content['total_steps']
                curr_step = step_content['step']
                gt_bbox = list(target_blocks.values())[0]
                if not gt_bbox:
                    gt_bbox = []
                else: 
                    gt_bbox = gt_bbox[0]
                    gt_bbox = [int(gt_bbox[0]), int(gt_bbox[1]), int(gt_bbox[0] + gt_bbox[2]), int(gt_bbox[1] + gt_bbox[3])]
                img_file = f'{self.img_path}/{block_path}/{block_id}.png'

                # plot gt
                gt_img_file = f'{self.gt_img_path}/{anno_id}_{action_uid}.png'
                if not u.is_file_exist(gt_img_file): 
                    img = Image.open(img_file)
                    gt_img = img.copy()
                    if gt_bbox:
                        gt_bbox_expand = expand_box(gt_bbox, 10)
                        gt_img = u.draw_bounding_box(gt_img, *gt_bbox_expand, True)
                    gt_img.save(gt_img_file)
                
                gt_action_str = map_to_actions(gt_action_type, gt_bbox, gt_action_value)
                prompt = prompt_f.format(task = task, action_history = u.dumps(gt_action_desc_history))
                response, _ = self.mc.infer_prompt(
                    sys_prompt, 
                    prompt,
                    [gt_img_file]
                )
                
                n_candidate_action = 0
                candidate_action_cot = ''
                select_reason = ''
                prebuild_action_intention = ''
                try:
                    response_results = u.extract_text(response, '```json\n', '```')[0]
                    response_results = json.loads(response_results)
                    select_reason = response_results['reason']
                    prebuild_action_intention = response_results['Candidate action number 1']['intention']
                    del response_results['reason']
                    n_candidate_action = len(response_results)
                    candidate_action_cot = ''
                    for action_num, c_action in response_results.items():
                        intention = c_action['intention']
                        if intention.endswith('.'): intention = intention[:-1]
                        if intention[0].isupper(): intention = intention[0].lower() + intention[1:]
                        consequence = c_action['consequence']
                        if consequence.endswith('.'): consequence = consequence[:-1]
                        candidate_action_cot += f'{action_num} is {intention}. {consequence}.\n'
                except Exception as e:
                    DEBUG(e)
                    n_candidate_action = 0
                    candidate_action_cot = ''
                    select_reason = ''
                    prebuild_action_intention = ''

                format_anno = \
                {
                    'anno_id': anno_id,
                    'action_uid': action_uid,
                    'n_step': n_step,
                    'curr_step': curr_step,
                    "conversations": [
                        {
                            "from": "human",
                            "value": (len(img_sw) + 1) * '<image>'
                        },
                        {
                            "from": "gpt",
                            "value": ''
                        }
                    ],
                    "system": sys_prompt, 
                    "images": []
                }

                format_anno['conversations'][0]['value'] += user_prompt_f.format(
                    task, u.dumps(gt_action_desc_history)
                )
                cot_str = cot.format(
                    query_str = task,
                    n_candidate_action = n_candidate_action,
                    candidate_action_cot = candidate_action_cot,
                    select_reason = select_reason,
                    gt_action_str = gt_action_str
                )
                cot_str = cot_str.replace('..', '.')
                format_anno['conversations'][1]['value'] = gt_answer.format(
                    cot = cot_str,
                    action_description = prebuild_action_intention,
                    action = gt_action_str
                )

                img_name = u.get_name(img_file)
                img_rel_file = img_file.replace('/ossfs/workspace/kaiwen/', '').replace('/ossfs/workspace//kaiwen/', '').replace('/mnt/agent-s1/common/public/kevin//', '').replace('/mnt/agent-s1/common/public/kevin/', '')

                format_anno['images'] = copy.deepcopy(img_sw) + [img_rel_file]
                format_anno['img_name'] = img_name
                format_anno['gt_action_type'] = gt_action_type
                format_anno['prebuild_action_desc'] = copy.deepcopy(prebuild_action_intention)
                format_anno['gt_action_str'] = gt_action_str
                format_anno['gt_action_value'] = gt_action_value
                format_anno['gt_bbox'] = gt_bbox
                format_anno['gt_action_desc_history'] = copy.deepcopy(gt_action_desc_history)

                sg_anno.append(format_anno)
                img_sw.append(img_rel_file)
                prebuild_action_history.append(prebuild_action_intention)
            u.write_json(self.anno_file, sg_anno, encoding='utf-8')

if __name__ == "__main__":
    si = SCOTInfer(sys)
    si.infer()