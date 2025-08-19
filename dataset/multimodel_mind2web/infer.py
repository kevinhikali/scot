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
import copy
from llm_service.model_cache import ModelCache, ModelLoader
from llm_service.vllm_inferer import VLLMInferer
from dataset_utils import get_output_model_name
from infer_utils import map_gt_action, map_action_desc_history
import prompt_parser as pp

INFERER = 'vllm'

class MMMWReader():
    def __init__(self, split):
        NAS_PATH = u.get_nas()
        dataset_name = 'MultiModel_Mind2Web'
        dataset_path = f'{NAS_PATH}/gui_dataset/{dataset_name}/sample/'
        main_path = f'{dataset_path}/{split}'
        anno_file = f'{main_path}/block_sample.jsonl'
        self.annotations = u.read_json(anno_file)
        self.img_path = f'{main_path}/image_blocks/'

    def read(self):
        annos_dict = {}
        for i in range(len(self.annotations)):
            step_content = self.annotations[i]
            anno_id = step_content['annotation_id']
            curr_step = step_content['step']
            if anno_id not in annos_dict.keys(): annos_dict[anno_id] = {}
            annos_dict[anno_id][curr_step] = step_content
            annos_dict[anno_id] = {key: annos_dict[anno_id][key] for key in sorted(annos_dict[anno_id].keys())}

        annos_format = {}
        for anno_id in annos_dict.keys():
            img_history = []
            gt_action_history = []
            if anno_id not in annos_format.keys(): annos_format[anno_id] = {}
            for curr_step in annos_dict[anno_id].keys():
                step_content = annos_dict[anno_id][curr_step]
                n_step = step_content['total_steps']
                action_uid = step_content['action_uid']
                blocks_path = step_content['blocks_path']
                task = step_content['task']
                gt_action_type = step_content['operation']
                gt_action_description = '' # TODO
                ori_operation = step_content['ori_operation']
                gt_value = step_content['value']
                target_blocks = step_content['target_blocks']
                ori_bbox = step_content['bbox']
                block_id = list(target_blocks.keys())[0]
                gt_bbox = list(target_blocks.values())[0]
                if not gt_bbox: 
                    gt_bbox = [0, 0, 0, 0]
                    gt_click_point = [0, 0]
                else: 
                    gt_bbox = gt_bbox[0]
                    gt_bbox = [int(gt_bbox[0]), int(gt_bbox[1]), int(gt_bbox[0] + gt_bbox[2]), int(gt_bbox[1] + gt_bbox[3])]
                    gt_click_point = [(gt_bbox[0] + gt_bbox[2]) / 2, (gt_bbox[1] + gt_bbox[3]) / 2]
                img_file = f'{self.img_path}/{blocks_path}/{block_id}.png'

                gt_action = map_gt_action(gt_action_type, gt_click_point, gt_value)
                gt_action_desc_history = step_content['previous_actions']
                gt_action_desc_history = map_action_desc_history(gt_action_desc_history)

                if len(img_history) != len(gt_action_desc_history):
                    ERROR(f'len(img_history) {len(img_history)} != len(gt_action_desc_history) {len(gt_action_desc_history)}')

                step_data = {
                    'anno_id': anno_id,
                    'action_uid': action_uid,
                    'curr_step': curr_step,
                    'n_step': n_step,
                    'task': task,
                    'img_history': copy.deepcopy(img_history), # i larger means newer
                    'ori_operation': ori_operation,
                    'ori_bbox': ori_bbox,
                    'gt_action_history': copy.deepcopy(gt_action_history),
                    'gt_action_desc_history': copy.deepcopy(gt_action_desc_history),
                    'gt_action_description': gt_action_description,
                    'gt_action': gt_action,
                    'gt_action_type': gt_action_type,
                    'gt_bbox': gt_bbox,
                    'gt_type_value': gt_value,
                    'gt_click_point': gt_click_point,
                }
                annos_format[anno_id][action_uid] = step_data
                img_history.append(img_file)
                gt_action_history.append(gt_action)

        return annos_format

    def filter_empty_bbox(self):
        annos = self.read()
        for anno_id in annos.keys():
            for action_uid in annos[anno_id].keys():
                step_content = annos[anno_id][action_uid]
                # if anno_id == '1c2baca4-8c20-4e04-b6f6-90db4f565a72' and action_uid == 'cb945343-a0bc-48f8-a041-15161f73be1d':
                #     u.print_json(step_content)
                #     exit()

                gt_bbox = step_content['gt_bbox']
                cond = (not gt_bbox) or (gt_bbox[0] < 0.01 and gt_bbox[1] < 0.01 and gt_bbox[2] < 0.01 and gt_bbox[3] < 0.01) 
                if cond:
                    u.print_json(step_content)
                    exit()


class MMInfer():
    def __init__(self, config):
        INFO('job_name n_chunks chunk_index main_split model_rel_path model_name')
        cfg = config
        job_name = cfg[1]
        n_chunks = int(cfg[2])
        chunk_index = int(cfg[3])
        main_split = cfg[4]
        model_rel_path = cfg[5]
        model_name = cfg[6]
        INFO(f'{job_name} {n_chunks} {chunk_index} {main_split} {model_rel_path} {model_name}')

        NAS_PATH = u.get_nas()
        model_path = f'{NAS_PATH}/{model_rel_path}/{model_name}'
        dataset_name = 'MultiModel_Mind2Web'
        dataset_path = f'{NAS_PATH}/gui_dataset/{dataset_name}/sample/'
        output_main_path = f'{NAS_PATH}/gui_dataset/{dataset_name}/output/'
        u.mkdir(output_main_path)
        output_model_name = get_output_model_name(model_name)
        output_model_name = f'{output_model_name}_{INFERER}'
        model_output_path = f'{output_main_path}/{output_model_name}'
        # u.pl('*')
        # DEBUG(model_output_path)
        # u.pl('*')
        # exit()
        u.mkdir(model_output_path)
        output_path = f'{model_output_path}/{main_split}/'
        u.mkdir(output_path)
        saved_path = f'{NAS_PATH}/gui_dataset/{dataset_name}/results/{output_model_name}/{main_split}/'
        self.img_output_path = f'{output_path}/imgs/'
        u.mkdir(self.img_output_path)
        self.img_output_path = f'{output_path}/imgs/{main_split}/'
        u.mkdir(self.img_output_path)

        main_path = f'{dataset_path}/{main_split}'
        anno_file = f'{main_path}/block_sample.jsonl'
        self.annotations = u.read_json(anno_file)

        n_all = len(self.annotations)
        n_sep = int(n_all / n_chunks)
        if chunk_index == (n_chunks - 1):
            self.annotations = self.annotations[chunk_index * n_sep:]
        else:
            self.annotations = self.annotations[chunk_index * n_sep: (chunk_index + 1) * n_sep]

        self.img_path = f'{main_path}/image_blocks/'

        self.results = []
        self.results_cache = {}
        self.result_file = f'{output_path}/results_{n_chunks}_{chunk_index}.json'
        saved_result_file = f'{saved_path}/results_{n_chunks}_{chunk_index}.json'
        if u.is_file_exist(saved_result_file): 
            self.results = u.read_json(saved_result_file)
        else:
            if u.is_file_exist(self.result_file): 
                self.results = u.read_json(self.result_file)

        if self.results:
            u.pl()
            INFO('Using cache')
            u.pl()
            for i in range(len(self.results)):
                anno_id = self.results[i]['anno_id']
                action_uid = self.results[i]['action_uid']
                if anno_id not in self.results_cache.keys(): self.results_cache[anno_id] = {}
                self.results_cache[anno_id][action_uid] = self.results[i]

        if model_name.endswith('sw0'): self.sw = 0
        elif model_name.endswith('sw1'): self.sw = 1
        elif model_name.endswith('sw2'): self.sw = 2
        elif model_name.endswith('sw3'): self.sw = 3
        elif model_name.endswith('sw4'): self.sw = 4
        else: 
            INFO('Attention default sw = 0')
            self.sw = 0

        self.model_name = model_name

        if INFERER == 'vllm':
            self.mc = VLLMInferer('qwen2_5_vl', model_path)
        else:
            self.mc = ModelLoader(model_path)

    def infer(self, infer_mode: str, parse_response, args: dict):
        if infer_mode == 'vl2': self.sw = 99999

        annos_dict = {}
        for i in range(len(self.annotations)):
            step_content = self.annotations[i]
            anno_id = step_content['annotation_id']
            curr_step = step_content['step']
            if anno_id not in annos_dict.keys(): annos_dict[anno_id] = {}
            annos_dict[anno_id][curr_step] = step_content
            annos_dict[anno_id] = {key: annos_dict[anno_id][key] for key in sorted(annos_dict[anno_id].keys())}

        for anno_id in tqdm(annos_dict.keys()):
            img_history = []
            gt_action_history = []
            pred_action_history = []
            pred_action_desc_history = []
            for curr_step in tqdm(annos_dict[anno_id].keys(), anno_id):
                step_content = annos_dict[anno_id][curr_step]
                n_step = step_content['total_steps']
                action_uid = step_content['action_uid']
                blocks_path = step_content['blocks_path']
                task = step_content['task']
                gt_action_type = step_content['operation']
                gt_action_description = '' # TODO
                ori_operation = step_content['ori_operation']
                gt_value = step_content['value']
                target_blocks = step_content['target_blocks']
                ori_bbox = step_content['bbox']
                block_id = list(target_blocks.keys())[0]
                gt_bbox = list(target_blocks.values())[0]
                if not gt_bbox: 
                    gt_bbox = [0, 0, 0, 0]
                    gt_click_point = [0, 0]
                else: 
                    gt_bbox = gt_bbox[0]
                    gt_bbox = [int(gt_bbox[0]), int(gt_bbox[1]), int(gt_bbox[0] + gt_bbox[2]), int(gt_bbox[1] + gt_bbox[3])]
                    gt_click_point = [(gt_bbox[0] + gt_bbox[2]) / 2, (gt_bbox[1] + gt_bbox[3]) / 2]
                img_file = f'{self.img_path}/{blocks_path}/{block_id}.png'

                gt_action = map_gt_action(gt_action_type, gt_click_point, gt_value)
                gt_action_desc_history = step_content['previous_actions']
                gt_action_desc_history = map_action_desc_history(gt_action_desc_history)

                if len(img_history) != len(gt_action_desc_history):
                    ERROR(f'len(img_history) {len(img_history)} != len(gt_action_desc_history) {len(gt_action_desc_history)}')

                if self.sw == 0: img_files = [img_file]
                elif len(img_history) <= self.sw: img_files = img_history + [img_file]
                else: img_files = img_history[-self.sw:] + [img_file]

                if self.results_cache and anno_id in self.results_cache.keys() and action_uid in self.results_cache[anno_id].keys():
                    cache_content = self.results_cache[anno_id][action_uid]
                    if 'message' in cache_content.keys(): messages = cache_content['message']
                    else: messages = cache_content['messages']
                    raw_response = cache_content['raw_response']
                    sys_prompt = cache_content['sys_prompt']
                    prompt = cache_content['prompt']
                elif infer_mode == 'prompt':
                    sys_prompt = args['sys_prompt']
                    prompt = args['prompt'](task, gt_action_desc_history)
                    raw_response, messages = self.mc.infer_prompt(sys_prompt, prompt, img_files)
                elif infer_mode == 'vl2':
                    sys_prompt, prompt, messages = args['message_builder'](task, gt_action_desc_history, gt_action_history, img_files)
                    raw_response = self.mc.infer_vl2(messages)
                else:
                    sys_prompt, prompt, messages = args['message_builder'](task, gt_action_desc_history, img_files)
                    raw_response = self.mc.infer_messages(messages, img_files)

                res_pred = parse_response(raw_response)

                pred_action = res_pred['pred_action']
                pred_action_description = res_pred['pred_action_description']
                # pred_click_point = res_pred['pred_click_point']
                # res_img_file = f'{self.img_output_path}/{anno_id}_{action_uid}.png'
                # if not u.is_file_exist(res_img_file): 
                #     img = Image.open(img_file)
                #     if gt_bbox:
                #         res_img = img.copy()
                #         res_img = u.draw_point(img, *pred_click_point, True)
                #         gt_bbox_expand = expand_box(gt_bbox, 10)
                #         res_img = u.draw_bounding_box(res_img, *gt_bbox_expand, True)
                #         res_img.save(res_img_file)

                res = {
                    'anno_id': anno_id,
                    'action_uid': action_uid,
                    'curr_step': curr_step,
                    'n_step': n_step,
                    'sys_prompt': sys_prompt,
                    'task': task,
                    'prompt': prompt,
                    'messages': messages,
                    'img_history': copy.deepcopy(img_history), # i larger means newer
                    'img_files': img_files,
                    'raw_response': raw_response,
                    'gt_action_history': copy.deepcopy(gt_action_history),
                    'gt_action_desc_history': copy.deepcopy(gt_action_desc_history),
                    'gt_action_description': gt_action_description,
                    'gt_action': gt_action,
                    'gt_action_type': gt_action_type,
                    'gt_bbox': gt_bbox,
                    'gt_type_value': gt_value,
                    'gt_click_point': gt_click_point,
                    'pred_action_history': copy.deepcopy(pred_action_history),
                    'pred_action_desc_history': copy.deepcopy(pred_action_desc_history),
                }
                res.update(res_pred)
                self.results.append(res)

                img_history.append(img_file)
                gt_action_history.append(gt_action)
                pred_action_history.append(pred_action)
                pred_action_desc_history.append(pred_action_description)
                u.write_json(self.result_file, self.results)

if __name__ == "__main__":
    mmi = MMInfer(sys.argv)
    model_name = sys.argv[6]

    if 'bbox' in model_name and '_pu_' in model_name:
        INFO('bbox pu')
        mmi.infer(
            'prompt', 
            pp.pu_bbox_parse_response, 
            {'sys_prompt': pp.sys_prompt, 'prompt': pp.pu_bbox_prompt_f.format}
        )
    elif 'pxxt' in model_name and '_pu_' in model_name:
        INFO('pxxt pu')
        mmi.infer(
            'prompt', 
            pp.pu_bbox_parse_response, 
            {'sys_prompt': pp.sys_prompt, 'prompt': pp.pu_bbox_prompt_f.format}
        )
    elif 'bbox' in model_name and 'd250624' in model_name:
        INFO('bbox d250624')
        mmi.infer(
            'prompt', 
            pp.parse_response_bbox_250610, 
            {'sys_prompt': pp.sys_prompt, 'prompt': pp.ori_bbox_prompt_f.format}
        )
    elif 'pxxt' in model_name and 'd250624' in model_name:
        INFO('pxxt d250624')
        mmi.infer(
            'prompt', 
            pp.parse_response_pxxt_250610, 
            {'sys_prompt': pp.sys_prompt, 'prompt': pp.ori_pxxt_prompt_f.format}
        )
    elif 'bbox' in model_name and 'd250714' in model_name:
        INFO('bbox d250714')
        mmi.infer(
            'prompt', 
            pp.parse_response_bbox_250714, 
            {'sys_prompt': pp.sys_prompt, 'prompt': pp.prompt_bbox_250714.format}
        )
    elif 'pxxt' in model_name and 'd250714' in model_name:
        INFO('pxxt d250714')
        mmi.infer(
            'prompt', 
            pp.parse_response_pxxt_250714, 
            {'sys_prompt': pp.sys_prompt, 'prompt': pp.prompt_pxxt_250714.format}
        )