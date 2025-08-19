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
import re
import json
from model_cache import ModelCache, ModelLoader
import copy
from infer_bbox import map_action_history

sys_prompt = '''You are a helpful assistant.'''

def extract_arguments(text):
    pattern = r'\{"name":\s*"computer_use",\s*"arguments":\s*(\{.*?\})\}'
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches: return None
    res = matches[0]
    res = '{"name": "computer_use", "arguments": ' + res + '}'
    return res

if __name__ == "__main__":
    job_name = sys.argv[1]
    n_chunks = int(sys.argv[2])
    chunk_index = int(sys.argv[3])
    main_split = sys.argv[4]
    model_rel_path = sys.argv[5]
    model_name = sys.argv[6]

    if u.get_os() == 'mac': NAS_PATH = f'{u.get_home()}/data/'
    else: NAS_PATH = '/mnt/agent-s1/common/public/kevin/'

    model_path = f'{NAS_PATH}/{model_rel_path}/{model_name}'
    if '/checkpoint-' in model_name: model_name = model_name.replace('/checkpoint-', '_ck_')
    dataset_name = 'MultiModel_Mind2Web'
    dataset_path = f'{NAS_PATH}/gui_dataset/{dataset_name}/sample/'
    output_main_path = f'{NAS_PATH}/gui_dataset/{dataset_name}/output/'
    u.mkdir(output_main_path)
    model_output_path = f'{output_main_path}/{model_name}_{job_name}'
    u.mkdir(model_output_path)
    output_path = f'{model_output_path}/{main_split}/'
    u.mkdir(output_path)
    saved_path = f'{NAS_PATH}/gui_dataset/{dataset_name}/results/{model_name}_{job_name}/{main_split}/'
    img_output_path = f'{output_path}/imgs/'
    u.mkdir(img_output_path)

    mc = ModelLoader(model_path)

    main_path = f'{dataset_path}/{main_split}'
    anno_file = f'{main_path}/block_sample.jsonl'
    annotations = u.read_json(anno_file)

    n_all = len(annotations)
    n_sep = int(n_all / n_chunks)
    if chunk_index == (n_chunks - 1):
        annotations = annotations[chunk_index * n_sep:]
    else:
        annotations = annotations[chunk_index * n_sep: (chunk_index + 1) * n_sep]

    img_path = f'{main_path}/image_blocks/'

    results = []
    results_cache = {}
    result_file = f'{output_path}/results_{n_chunks}_{chunk_index}.json'
    saved_result_file = f'{saved_path}/results_{n_chunks}_{chunk_index}.json'
    if u.is_file_exist(saved_result_file): 
        results = u.read_json(saved_result_file)
    else:
        if u.is_file_exist(result_file): results = u.read_json(result_file)

    if results:
        for i in range(len(results)):
            anno_id = results[i]['anno_id']
            action_uid = results[i]['action_uid']
            if anno_id not in results_cache.keys(): results_cache[anno_id] = {}
            results_cache[anno_id][action_uid] = results[i]

    for step_content in tqdm(annotations):
        anno_id = step_content['annotation_id']
        action_uid = step_content['action_uid']
        block_path = step_content['blocks_path']
        task = step_content['task']
        operation = step_content['operation']
        ori_operation = step_content['ori_operation']
        gt_value = step_content['value']
        target_blocks = step_content['target_blocks']
        ori_bbox = step_content['bbox']
        block_id = list(target_blocks.keys())[0]
        gt_bbox = list(target_blocks.values())[0]
        if not gt_bbox: gt_bbox = []
        else: 
            gt_bbox = gt_bbox[0]
            gt_bbox = [int(gt_bbox[0]), int(gt_bbox[1]), int(gt_bbox[0] + gt_bbox[2]), int(gt_bbox[1] + gt_bbox[3])]
        img_file = f'{img_path}/{block_path}/{block_id}.png'
        
        gt_action_history = step_content['previous_actions']
        gt_action_history = map_action_history(gt_action_history)

        n_step = step_content['total_steps']
        curr_step = step_content['step']

        img_sw = []
        action_history = []

        img_name = u.get_name(img_file)
        img_rel_file = img_file.replace('/ossfs/workspace/kaiwen/', '').replace('/ossfs/workspace//kaiwen/', '').replace('/mnt/agent-s1/common/public/kevin//', '').replace('/mnt/agent-s1/common/public/kevin/', '')

        response = ''
        if results_cache and anno_id in results_cache.keys() and action_uid in results_cache[anno_id].keys():
            response = results_cache[anno_id][action_uid]['raw_response']
            message = results_cache[anno_id][action_uid]['message']
        else:
            response, message = mc.inference_cua(img_file, task)

        response_results = ''
        try: response_results = u.extract_text(response, '<tool_call>', '\n</tool_call>')[0]
        except: pass
        if not response_results:
            try: response_results = extract_arguments(response)
            except: pass

        pred_bbox = []
        try: 
            response_results = json.loads(response_results)
            pred_action_type = response_results['arguments']['action']
            pred_click_point = response_results['arguments']['coordinate']
            pred_value = ''
        except:
            pred_action_type = 'UNKNOWN'
            pred_click_point = [0, 0]
            pred_value = ''

        if 'click' in pred_action_type:
            pred_ac_type = 'CLICK'
            pred_value = ''
        elif 'select' in pred_action_type:
            pred_ac_type = 'SELECT'
            pred_value = ''
        elif 'type' in pred_action_type:
            DEBUG(response_results)
            pred_ac_type = 'TYPE'
            pred_value = ''
        else:
            pred_ac_type = 'UNKNOWN'
            pred_value = ''

        res_img_file = f'{img_output_path}/{anno_id}_{action_uid}.png'
        if not u.is_file_exist(res_img_file): 
            img = Image.open(img_file)
            if gt_bbox:
                res_img = img.copy()
                res_img = u.draw_point(img, *pred_click_point, True)
                res_img = u.draw_bounding_box(res_img, *gt_bbox, True)
                res_img.save(res_img_file)

        res = {
            'anno_id': anno_id,
            'action_uid': action_uid,
            'curr_step': curr_step,
            'n_step': n_step,
            'sys_prompt': sys_prompt,
            'task': task,
            'prompt': 'cua',
            'message': message,
            'gt_action_history': gt_action_history,
            'pred_action_history': action_history,
            'img_file': img_file,
            'raw_response': response,
            'action_description': '',
            'pred_action': '',
            'pred_action_type': pred_ac_type,
            'pred_bbox': '',
            'pred_type_value': pred_value,
            'pred_click_point': pred_click_point,
            'gt_operation': operation,
            'gt_bbox': gt_bbox,
            'gt_type_value': gt_value
        }
        results.append(res)
        u.write_json(result_file, results)
