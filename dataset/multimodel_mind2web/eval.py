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
import collections
import numpy as np
from tqdm import tqdm
import os
from PIL import Image
import string
import re
import argparse
from prompt_parser import parse_response_bbox_250610 as bbox_pr
from prompt_parser import parse_response_pxxt_250610 as pxxt_pr
from prompt_parser import pu_bbox_parse_response as bbox_pr_pure
from prompt_parser import pu_pxxt_parse_response as pxxt_pr_pure
from infer_uitars import parse_response as uitars_pr

def calculate_f1(pred, label):
    pred = set(pred.lower().strip().split())
    label = set(label.lower().strip().split())
    # remove punctuation
    pred = set([x for x in pred if x not in string.punctuation])
    label = set([x for x in label if x not in string.punctuation])
    if len(pred) == 0 and len(label) == 0:
        return 1
    if len(pred) == 0 or len(label) == 0:
        return 0

    tp = len(pred & label)
    fp = len(pred - label)
    fn = len(label - pred)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision == 0 or recall == 0:
        return 0
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def is_output_inside_bbox(bboxes, output, scale):
    output_x, output_y = output
    output_x /= scale
    output_y /= scale

    for bbox in bboxes:
        bbox_x, bbox_y, bbox_width, bbox_height = bbox
        if bbox_x <= output_x <= bbox_x + bbox_width and bbox_y <= output_y <= bbox_y + bbox_height:
            return True, (output_x, output_y)
    return False, (output_x, output_y)

def extract_coordinates(operation, image_path):
    # extract for cogagent output
    tap_match = re.search(r'tap\s*\[\[(\d+),(\d+)\]\]', operation, re.IGNORECASE)
    box_match = re.search(r'\[\[(\d+),(\d+),(\d+),(\d+)\]\]', operation)

    image = Image.open(image_path)
    width, height = image.size
    
    if tap_match:
        x, y = map(int, tap_match.groups())
        x = int(width * (x / 1000))
        y = int(height * (y / 1000))
        return (x, y)
    elif box_match:
        x1, y1, x2, y2 = map(int, box_match.groups())
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        center_x = int(width * (center_x / 1000))
        center_y = int(height * (center_y / 1000))
        return (center_x, center_y)
    else:
        raise ValueError("Operation format not recognized", operation)

def get_metrics_with_prediction(sample_data, plan_data, ans_data):
    all_element_acc = []
    all_operation_f1 = []
    all_step_acc = []
    sample_to_website = {}
    
    for sample in sample_data:
        annotation_id = sample['annotation_id']
        action_uid = sample['action_uid']
        sample_id = f"{annotation_id}_{action_uid}"
        
        sample_to_website[annotation_id] = sample["website"]
        
        # Get planner data
        planner_entry = next((item for item in plan_data if item['annotation_id'] == annotation_id and item['action_uid'] == action_uid), None)
        if planner_entry:
            gpt_action = planner_entry["gpt_action"].lower()
            gpt_value = planner_entry["gpt_value"].lower()
            if gpt_value == "none":
                pred_action = gpt_action
            else:
                pred_action = f"{gpt_action} {gpt_value}"
        else:
            pred_action = ""
        
        # Get ans data
        if ans_data[0].get("id", ""):
            ans_entry = next((item for item in ans_data if item['id'] == sample_id), None)
        else:
            ans_entry = planner_entry
        if ans_entry:
            output = ans_entry.get("output", "")
            ans_block = ans_entry.get("ans_block", "")
            if not output: output = (0, 0)
            
            bboxes = ans_entry.get("bbox", [])
            scale = ans_entry.get("scale", 1.0)
            
            correct, coords = is_output_inside_bbox(bboxes, output, scale)
            all_element_acc.append([1 if correct else 0, annotation_id])
        else:
            all_element_acc.append([0, annotation_id])
        
        current_action = (sample["operation"], sample["value"])
        f1_score = calculate_f1(pred_action, current_action[0]+" "+current_action[1])
        all_operation_f1.append([f1_score, annotation_id])
        all_step_acc.append([1 if (all_operation_f1[-1][0]==1 and all_element_acc[-1][0]==1) else 0, annotation_id])
    
    total_steps = {sample['annotation_id']: sample['total_steps'] for sample in sample_data}
    current_steps = collections.defaultdict(int)
    for _, annotation_id in all_element_acc:
        current_steps[annotation_id] += 1
    for annotation_id, steps in total_steps.items():
        while current_steps[annotation_id] < steps:
            all_element_acc.append([0, annotation_id])
            all_operation_f1.append([0, annotation_id])
            all_step_acc.append([0, annotation_id])
            current_steps[annotation_id] += 1
    
    macro_element_acc = collections.defaultdict(list)
    macro_operation_f1 = collections.defaultdict(list)
    macro_step_acc = collections.defaultdict(list)
    for x in all_element_acc:
        macro_element_acc[x[1]].append(x[0])
    for x in all_operation_f1:
        macro_operation_f1[x[1]].append(x[0])
    for x in all_step_acc:
        macro_step_acc[x[1]].append(x[0])
    
    error_ratio = collections.defaultdict(int)
    acc_per_website = collections.defaultdict(list)
    for annotation_id, x in macro_step_acc.items():
        acc_per_website[sample_to_website[annotation_id]].append(np.mean(x))
        error_count = len([y for y in x if y == 0])
        if error_count <= 3:
            error_ratio[error_count] += 1
        else:
            error_ratio[">3"] += 1
    
    acc_per_website = {k: (np.mean(v), len(v)) for k, v in acc_per_website.items()}
    error_ratio = {k: v/len(macro_element_acc) for k, v in error_ratio.items()}
    macro_element_acc = np.mean([np.mean(x) for x in macro_element_acc.values()])
    macro_operation_f1 = np.mean([np.mean(x) for x in macro_operation_f1.values()])
    macro_step_acc = np.mean([np.mean(x) for x in macro_step_acc.values()])

    all_metrics = {
        "element_acc": np.mean([x[0] for x in all_element_acc]),
        "operation_f1": np.mean([x[0] for x in all_operation_f1]),
        "step_acc": np.mean([x[0] for x in all_step_acc]),
        "macro_element_acc": macro_element_acc,
        "macro_operation_f1": macro_operation_f1,
        "macro_step_acc": macro_step_acc,
        "error_ratio": error_ratio,
        "acc_per_website": acc_per_website,
    }

    return all_metrics, [x[0] for x in all_step_acc]

SKIP_FOLDERS = [
    'Qw2.5VL72BIns_cua',
    'Qw2.5VL7BIns_cua',
]

ORI_FOLDERS = [
    'Qw2.5VL72BIns_bbox_wo_hint',
    'Qw2.5VL72BIns_pt_prompt_v1',
    'Qw2.5VL72BIns_pt_prompt_v2',
    'Qw2.5VL7BIns_bbox_wo_hint',
    'Qw2.5VL7BIns_pt_prompt_v1',
    'Qw2.5VL7BIns_pt_prompt_v2',
]

PURE_FOLDERS = [
    'Qw2.5VL7BIns_bbox_sw0_pu_d250618_sft_ck_180_vllm',
    'Qw2.5VL7BIns_pxxt_sw0_pu_d250618_sft_ck_180_vllm',
]

def get_parser(folder_name):
    if folder_name in PURE_FOLDERS and '_bbox_' in folder_name:
        INFO('parser bbox_pr_pure')
        return bbox_pr_pure
    elif folder_name in PURE_FOLDERS and ('_pxxt_' in folder_name or '_pt_' in folder_name):
        INFO('parser pt_pr_pure')
        return pxxt_pr_pure
    elif '_bbox_' in folder_name:
        INFO('parser bbox_pr')
        return bbox_pr
    elif ('_pxxt_' in folder_name or '_pt_' in folder_name or '_ptv2_' in folder_name):
        INFO('parser pt_pr')
        return pxxt_pr
    elif 'UI-TARS' in folder_name:
        INFO('parser uitars_pr')
        return uitars_pr
    else:
        raise ValueError("Folder name not recognized", folder_name)

if __name__ == "__main__":
    INFO('result_folder')
    result_folder = sys.argv[1]

    flush = 0
    main_path = f'{u.get_nas()}/gui_dataset/MultiModel_Mind2Web/'
    sample_path = f'{main_path}/sample/'
    result_path = f'{main_path}/{result_folder}/'

    models = u.list_folders(result_path, False)
    models = [a for a in models if 'z_deprecated' not in a]
    splits = ['cross_task', 'cross_website', 'cross_domain']

    sorted_metrics_str = []
    for model in tqdm(models):
        if model in SKIP_FOLDERS: continue
        parser = get_parser(model)
        model_output_path = f'{result_path}/{model}'
        metrics_file = f'{model_output_path}/metrics.json'
        sr_file = f'{model_output_path}/sr.json'

        errors = {}
        error_file = f'{model_output_path}/error_msg.json'
        if not flush and u.is_file_exist(error_file): errors = u.read_json(error_file)

        model_metrics = {}
        model_sr = {}
        if u.is_file_exist(metrics_file) and u.is_file_exist(sr_file) and not flush:
            model_metrics = u.read_json(metrics_file)
            model_sr = u.read_json(sr_file)
        else:
            for split in tqdm(splits, model):
                sample_file = f'{sample_path}/{split}/block_sample.jsonl'
                sample_data = u.read_json(sample_file)
                result_file_path = f'{model_output_path}/{split}/'
                result_files = u.list_files(result_file_path, True)
                pattern = re.compile(r"^results_\d+_\d+\.json$")
                result_files = [a for a in result_files if pattern.match(u.get_name(a))]

                results = {}
                for result_file in result_files:
                    result_info = u.read_json(result_file)
                    for item in result_info:
                        anno_id = item['anno_id'] 
                        action_uid = item['action_uid'] 
                        if anno_id not in results.keys(): results[anno_id] = {}
                        results[anno_id][action_uid] = item

                ans = []
                for i in tqdm(range(len(sample_data)), f'{model}_{split}'):
                    item = sample_data[i]
                    anno_id = item['annotation_id']
                    action_uid = item['action_uid']
                    try: all_output = results[anno_id][action_uid]
                    except Exception as e: 
                        ERROR(f'{model} {e}')
                        exit()
                    block_id = list(sample_data[i]['target_blocks'].keys())[0]
                    sample_data[i]['ans_block'] = block_id

                    raw_response = all_output['raw_response']
                    gt_bbox = all_output['gt_bbox']
                    pr_results = parser(raw_response)
                    pred_cp = pr_results['pred_click_point']
                    cond0 = (model not in ORI_FOLDERS)
                    cond1 = (not pred_cp) or (pred_cp[0] < 0.01 and pred_cp[1] < 0.01) 
                    cond2 = (not gt_bbox) or (gt_bbox[0] < 0.01 and gt_bbox[1] < 0.01 and gt_bbox[2] < 0.01 and gt_bbox[3] < 0.01) 
                    if cond0 and (cond1 or cond2):
                        if split not in errors.keys(): errors[split] = {}
                        if anno_id not in errors[split].keys(): errors[split][anno_id] = {}
                        if action_uid not in errors[split][anno_id].keys():
                            error_msg = {
                                'raw_response': raw_response,
                                'pr_results': pr_results,
                                'all_output': all_output
                            }
                            errors[split][anno_id][action_uid] = error_msg

                    sample_data[i]['gpt_action'] = pr_results['pred_action_type']
                    sample_data[i]['gpt_value'] = pr_results['pred_type_value']
                    description = pr_results['pred_action_description']
                    sample_data[i]['description'] = description
                    pred_click_point = pr_results['pred_click_point']
                    gt_bbox_xyxy = all_output['gt_bbox']

                    # bbox_xyxy = result_output['pred_bbox']
                    if gt_bbox_xyxy:
                        gt_bbox_xywh = [
                            gt_bbox_xyxy[0], 
                            gt_bbox_xyxy[1], 
                            gt_bbox_xyxy[2] - gt_bbox_xyxy[0], 
                            gt_bbox_xyxy[3] - gt_bbox_xyxy[1]
                        ]
                    else:
                        gt_bbox_xywh = [0, 0, 0, 0]

                    pred = results[anno_id][action_uid]
                    gt_action_type = item['operation']
                    pred_action_type = pred['pred_action_type']
                    if gt_action_type == pred_action_type: 
                        results[anno_id][action_uid]['action_type_the_same'] = 1
                    else:
                        results[anno_id][action_uid]['action_type_the_same'] = 0

                    gt_type_value = item['value']
                    pred_type_value = pred['pred_type_value']
                    if gt_type_value == '' or pred_type_value == gt_type_value:
                        results[anno_id][action_uid]['type_value_the_same'] = 1
                    else:
                        results[anno_id][action_uid]['type_value_the_same'] = 0

                    gt_bbox = gt_bbox_xyxy
                    pred_click_point = pred['pred_click_point']

                    if not gt_bbox: 
                        results[anno_id][action_uid]['empty_gt_bbox'] = 1
                        results[anno_id][action_uid]['pred_format_error'] = 0
                        results[anno_id][action_uid]['click_point_in_bbox'] = 0
                    elif not pred_click_point:
                        results[anno_id][action_uid]['empty_gt_bbox'] = 0
                        results[anno_id][action_uid]['pred_format_error'] = 1
                        results[anno_id][action_uid]['click_point_in_bbox'] = 0
                    else: 
                        results[anno_id][action_uid]['empty_gt_bbox'] = 0
                        results[anno_id][action_uid]['pred_format_error'] = 0
                        if (gt_bbox[0] <= pred_click_point[0] <= gt_bbox[2]) and (gt_bbox[1] <= pred_click_point[1] <= gt_bbox[3]): 
                            results[anno_id][action_uid]['click_point_in_bbox'] = 1
                        else:
                            results[anno_id][action_uid]['click_point_in_bbox'] = 0

                    ans_info = {
                        'id': item['blocks_path'],
                        'image': f'''{item['blocks_path']}/{block_id}.png''',
                        'bbox': [gt_bbox_xywh],
                        'description': description,
                        'scale': 1.00,
                        'output': pred_click_point,
                    }
                    ans.append(ans_info)
                
                plan_file = f'{result_file_path}/plan.jsonl'
                u.write_jsonl(plan_file, sample_data)
                ans_file = f'{result_file_path}/ans.jsonl'
                u.write_jsonl(ans_file, ans)
                result_file = f'{result_file_path}/results.json'
                u.write_json(result_file, results, encoding='utf-8')

                plan_data = u.read_json(plan_file)
                ans_data = u.read_json(ans_file)

                metrics, step_accs = get_metrics_with_prediction(sample_data, plan_data, ans_data)

                model_metrics[split] = metrics
                model_sr[split] = step_accs

            if errors: u.write_json(error_file, errors, encoding = 'utf-8')
            u.write_json(metrics_file, model_metrics)
            u.write_json(sr_file, model_sr)

        sr_avg = np.mean(model_sr['cross_task'] + model_sr['cross_website'] + model_sr['cross_domain'])
        model_str = model.replace('_', ' ')
        if 'SCOT' in model_str:
            model_str = f'''\\textbf{{{model_str}}}'''
        if 'DGRPO' in model_str:
            model_str = f'''\\textbf{{{model_str}}}'''

        metrics_str = \
f'''{model_str} & {model_metrics['cross_task']['element_acc']*100:.1f} & {model_metrics["cross_task"]['operation_f1']*100:.1f} & {model_metrics['cross_task']['step_acc']*100:.1f} & {model_metrics['cross_website']['element_acc']*100:.1f} & {model_metrics["cross_website"]['operation_f1']*100:.1f} & {model_metrics['cross_website']['step_acc']*100:.1f} & {model_metrics['cross_domain']['element_acc']*100:.1f} & {model_metrics["cross_domain"]['operation_f1']*100:.1f} & {model_metrics['cross_domain']['step_acc']*100:.1f} & {sr_avg*100:.1f} \\\\'''
        print(metrics_str)
        
        sorted_metrics_str.append([metrics_str, sr_avg])

    sorted_metrics_str = sorted(sorted_metrics_str, key=lambda x: x[1])
    u.pl()
    for a in sorted_metrics_str:
        print(a[0])