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
import copy
import pandas as pd
from tqdm import tqdm
import numpy as np
import re

def get_wo_image_token(s: str):
    s_ = copy.deepcopy(s)
    while s_.startswith('<image>\n'):
        s_ = s_[8:]
    return s_

def check_anno_file(data_json_file):
    if not u.is_file_exist(data_json_file): 
        WARN(f'Checking found file = {data_json_file} does not exist')
        return
    train_annos = u.read_json(data_json_file)
    for i in tqdm(range(len(train_annos)), 'check'):
        instruction = train_annos[i]['conversations'][0]['value']
        n_image_token = instruction.count('<image>')
        n_images = len(train_annos[i]['images'])
        if n_image_token != n_images:
            WARN(f'Checking found unequal image token with image count, file = {data_json_file}, n_anno = {i}')
            pure_s = get_wo_image_token(instruction)
            pure_s = pure_s.replace('<image>', '<img>')
            instruction_p = n_images * '<image>\n' + pure_s
            train_annos[i]['conversations'][0]['value'] = instruction_p
    json_path = u.get_path(data_json_file)
    filename = u.get_name(data_json_file, True)
    u.copy_file(data_json_file, f'{json_path}/{filename}_backup.json')
    u.write_json(data_json_file, train_annos, encoding = 'utf-8')

if __name__ == "__main__":
    train_folder = sys.argv[1]
    main_path = f'{u.get_nas()}/gui_dataset/{train_folder}/'
    files = u.list_files(main_path, True)
    pattern_train = re.compile(r"^train_\d+_\d+\.json$")
    pattern_val = re.compile(r"^val_\d+_\d+\.json$")
    train_files = [a for a in files if pattern_train.match(u.get_name(a))]
    val_files = [a for a in files if pattern_val.match(u.get_name(a))]

    train_anno_file = f'{main_path}/train.json'
    train_annos = []
    for train_file in train_files:
        train_annos += u.read_json(train_file)
    u.write_json(train_anno_file, train_annos, encoding='utf-8')

    val_anno_file = f'{main_path}/val.json'
    val_annos = []
    for val_file in val_files:
        val_annos += u.read_json(val_file)
    u.write_json(val_anno_file, val_annos, encoding='utf-8')

    json_files = [train_anno_file, val_anno_file]

    for json_file in json_files:
        output_path = f'{main_path}'
        ori_contents = u.read_json(json_file)
        N_SWs = [0]

        json_name = u.get_name(json_file, True)

        # for bbox
        n_bbox_skip = 0
        for N_SW in tqdm(N_SWs):
            output_file = f'{output_path}/{json_name}_bbox_SW{N_SW}.json'
            contents = []
            for step_content in ori_contents:
                if 'val' not in json_name and not step_content['gt_bbox']: 
                    n_bbox_skip += 1
                    continue
                contents.append(step_content)
            for i in range(len(contents)):
                answer = contents[i]['conversations'][1]['value']
                gt_bbox = contents[i]['gt_bbox']
                gt_click_point = []
                if gt_bbox:
                    pattern = r'<\|box_start\|>\[\d+, \d+, \d+, \d+\]<\|box_end\|>'
                    matches = [(m.start(), m.end()) for m in re.finditer(pattern, answer)]
                    gt_click_point = [int((gt_bbox[0] + gt_bbox[2]) / 2), int((gt_bbox[1] + gt_bbox[3]) / 2)]
                contents[i]['gt_click_point'] = gt_click_point
                while contents[i]['conversations'][0]['value'].startswith((N_SW+2)*'<image>'):
                    contents[i]['conversations'][0]['value'] = contents[i]['conversations'][0]['value'].replace((N_SW+2)*'<image>', (N_SW+1)*'<image>')
                while len(contents[i]['images']) >= (N_SW + 2):
                    contents[i]['images'] = contents[i]['images'][1:]
                contents[i]['gt_str'] = json.dumps(contents[i])
            u.write_json(output_file, contents, encoding='utf-8')
            check_anno_file(output_file)
        INFO(f'n_bbox_skip = {n_bbox_skip}')

        # for point
        n_pt_skip = 0
        for N_SW in tqdm(N_SWs):
            output_file = f'{output_path}/{json_name}_pxxt_SW{N_SW}.json'
            contents = []
            for step_content in ori_contents:
                if 'val' not in json_name and not step_content['gt_bbox']: 
                    n_pt_skip += 1
                    continue
                contents.append(step_content)
            for i in range(len(contents)):
                prompt = contents[i]['conversations'][0]['value']
                prompt = prompt.replace('x1, y1, x2, y2', 'x, y')
                prompt = prompt.replace('in box x, y', 'at point x, y')
                prompt = prompt.replace(',.', ',')
                prompt = prompt.replace('.,', ',')
                contents[i]['conversations'][0]['value'] = prompt

                answer = contents[i]['conversations'][1]['value']
                gt_bbox = contents[i]['gt_bbox']
                gt_click_point = []
                if gt_bbox:
                    pattern = r'<\|box_start\|>\[\d+, \d+, \d+, \d+\]<\|box_end\|>'
                    matches = [(m.start(), m.end()) for m in re.finditer(pattern, answer)]
                    gt_click_point = [int((gt_bbox[0] + gt_bbox[2]) / 2), int((gt_bbox[1] + gt_bbox[3]) / 2)]
                    point_answer = ''
                    last_end = 0
                    for match in matches:
                        si = match[0]
                        ei = match[1]
                        point_answer += answer[last_end:si]
                        point_answer += f'<|box_start|>[{gt_click_point[0]}, {gt_click_point[1]}]<|box_end|>'
                        last_end = ei
                    point_answer += answer[last_end:]
                    point_answer = point_answer.replace(',.', ',')
                    point_answer = point_answer.replace('.,', ',')
                    contents[i]['conversations'][1]['value'] = point_answer

                    gt_action_str = contents[i]['gt_action_str']
                    matches = [(m.start(), m.end()) for m in re.finditer(pattern, gt_action_str)]
                    point_gt_action_str = ''
                    for match in matches:
                        si = match[0]
                        ei = match[1]
                        point_gt_action_str += gt_action_str[0:si]
                        point_gt_action_str += f'<|box_start|>[{gt_click_point[0]}, {gt_click_point[1]}]<|box_end|>'
                        point_gt_action_str += gt_action_str[ei:]
                    contents[i]['conversations'][1]['value'] = point_answer
                    contents[i]['gt_action_str'] = point_gt_action_str

                contents[i]['gt_click_point'] = gt_click_point

                while contents[i]['conversations'][0]['value'].startswith((N_SW+2)*'<image>'):
                    contents[i]['conversations'][0]['value'] = contents[i]['conversations'][0]['value'].replace((N_SW+2)*'<image>', (N_SW+1)*'<image>')
                while len(contents[i]['images']) >= (N_SW + 2):
                    contents[i]['images'] = contents[i]['images'][1:]
                contents[i]['gt_str'] = json.dumps(contents[i])
            u.write_json(output_file, contents, encoding='utf-8')
            check_anno_file(output_file)
        INFO(f'n_pt_skip = {n_pt_skip}')