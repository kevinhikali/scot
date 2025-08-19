import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
up_dir = parent_dir
for i in range(3):
    sys.path.append(up_dir)
    up_dir = os.path.dirname(up_dir)
from kutils import DEBUG, INFO, WARN, ERROR
import utils as u
import os
import json
from PIL import Image
from tqdm import tqdm
from llm_service.model_cache import ModelCache, ModelLoader, SimpleModel
from llm_service.vllm_inferer import VLLMInferer
from llm_service.longchen_requestor import request_longchen_gpt4o
from dataset.multimodel_mind2web.infer import map_action_desc_history, map_gt_action

prompt_f = \
'''You will be provided with a screenshot of a Graphical User Interface (GUI), where the ground truth action is depicted by the red bounding box, and a pred action from an AI assistant. Your job is to determine whether the intention and operation of the gt image and pred action are the same. Give me the reason of your determination.

Pred action by the AI assistant:
{}

Output your results in the following json format, where "result" field should be one of "same" or "not the same":
```json
{{
    "reason": "",
    "result": ""
}}
```'''

class MMMWAna():
    def __init__(self, config):
        INFO('n_chunks chunk_index main_split result_folder inferer')
        cfg = config
        n_chunks = int(cfg[1])
        chunk_index = int(cfg[2])
        main_split = cfg[3]
        result_folder = cfg[4]
        inferer = cfg[5]
        INFO(f'{n_chunks} {chunk_index} {main_split} {result_folder} {inferer}')

        dataset_name = 'MultiModel_Mind2Web'
        dataset_path = f'{u.get_nas()}/gui_dataset/{dataset_name}/'
        result_path = f'{dataset_path}/results/'
        output_main_path = f'{dataset_path}/ana/'
        u.mkdir(output_main_path)
        result_output_path = f'{output_main_path}/{result_folder}_{inferer}'
        u.mkdir(result_output_path)
        output_path = f'{result_output_path}/{main_split}/'
        u.mkdir(output_path)
        self.img_output_path = f'{output_path}/imgs/'
        u.mkdir(self.img_output_path)
        self.gt_img_path = f'{dataset_path}/gt_imgs/{main_split}/'

        main_path = f'{dataset_path}/sample/{main_split}'
        anno_file = f'{main_path}/block_sample.jsonl'
        desc_file = f'{main_path}/desc.json'
        self.annotations = u.read_json(anno_file)
        self.descs = u.read_json(desc_file)
        result_file = f'{result_path}/{result_folder}/{main_split}/results.json'
        self.results = u.read_json(result_file)

        n_all = len(self.annotations)
        n_sep = int(n_all / n_chunks)
        if chunk_index == (n_chunks - 1):
            self.annotations = self.annotations[chunk_index * n_sep:]
        else:
            self.annotations = self.annotations[chunk_index * n_sep: (chunk_index + 1) * n_sep]
        self.img_path = f'{main_path}/image_blocks/'

        self.ana = {}
        self.ana_file = f'{output_path}/ana_{n_chunks}_{chunk_index}.json'
        if u.is_file_exist(self.ana_file): 
            self.ana = u.read_json(self.ana_file)
            u.pl()
            INFO('Using cache')
            u.pl()

        if inferer == 'vllm':
            model_path = f'{u.get_nas()}/model/Qwen2.5-VL-72B-Instruct/'
            self.mc = VLLMInferer('qwen2_5_vl', model_path)
        elif inferer == 'transformers':
            model_path = f'{u.get_nas()}/model/Qwen2.5-VL-72B-Instruct/'
            self.mc = ModelLoader(model_path)
        elif inferer == 'gpt-4o':
            self.mc = SimpleModel()
        else:
            raise ValueError('Error inferer ', inferer)

    def infer(self):
        annos_dict = {}
        for i in range(len(self.annotations)):
            step_content = self.annotations[i]
            anno_id = step_content['annotation_id']
            curr_step = step_content['step']
            if anno_id not in annos_dict.keys(): annos_dict[anno_id] = {}
            annos_dict[anno_id][curr_step] = step_content
            annos_dict[anno_id] = {key: annos_dict[anno_id][key] for key in sorted(annos_dict[anno_id].keys())}

        for anno_id in tqdm(annos_dict.keys()):
            if anno_id not in self.ana.keys(): self.ana[anno_id] = {}
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
                gt_action_desc_history = step_content['previous_actions']
                gt_action_desc_history = map_action_desc_history(gt_action_desc_history)

                gt_img_file = f'{self.gt_img_path}/{anno_id}_{action_uid}.png'
                gt_action = map_gt_action(gt_action_type, gt_bbox, gt_value)
                result = self.results[anno_id][action_uid]
                pred_click_point = result['pred_click_point']
                pred_action = result['pred_action']
                pred_action_desc = result['pred_action_description']
                gpt_action_desc = self.descs[anno_id][action_uid]['action_description']

                # res_img_file = f'{self.img_output_path}/{anno_id}_{action_uid}.png'
                # if not u.is_file_exist(res_img_file): 
                #     img = Image.open(gt_img_file)
                #     res_img = img.copy()
                #     res_img = u.draw_point(res_img, *pred_click_point, True, color = 'blue')
                #     res_img = u.add_text_bottom(res_img, f'{task}\npred_{pred_action_desc}')
                #     res_img.save(res_img_file)

                action_type_the_same = result["action_type_the_same"]
                type_value_the_same = result["type_value_the_same"]
                empty_gt_bbox = result["empty_gt_bbox"]
                pred_format_error = result["pred_format_error"]
                click_point_in_bbox = result["click_point_in_bbox"]

                compare_result = {
                    'task': task,
                    "pred_click_point": pred_click_point,
                    "pred_action": pred_action,
                    "gt_action": gt_action,
                    "pred_action_desc": pred_action_desc,
                    "gpt_action_desc": gpt_action_desc,
                    "action_type_the_same": action_type_the_same,
                    "type_value_the_same": type_value_the_same,
                    "empty_gt_bbox": empty_gt_bbox,
                    "pred_format_error": pred_format_error,
                    "click_point_in_bbox": click_point_in_bbox,
                    "determine_reason": '',
                    "determine_result": '',
                    "determine_reliability": 0,
                    "error_msg": ''
                }

                if action_uid not in self.ana[anno_id].keys():
                    sys_prompt = 'You are a helpful assistant.'
                    response, _ = self.mc.infer_prompt(sys_prompt, prompt_f.format(pred_action_desc), [gt_img_file])

                    try: response_results = u.extract_text(response, '```json', '```')[0]
                    except Exception as e: 
                        compare_result['error_msg'] += f'{response}, error at extract text {e}\n'
                        self.ana[anno_id][action_uid] = compare_result
                        continue
                    try: response_results = json.loads(response_results)
                    except Exception as e: 
                        compare_result['error_msg'] += f'{response}, error at json loads {e}\n'
                        self.ana[anno_id][action_uid] = compare_result
                        continue
                    try: 
                        compare_result['determine_reason'] = response_results['reason']
                    except Exception as e: 
                        compare_result['error_msg'] += f'{response}, error at reason {e}\n'
                        self.ana[anno_id][action_uid] = compare_result
                        continue
                    try: 
                        compare_result['determine_result'] = response_results['result']
                    except Exception as e: 
                        compare_result['error_msg'] += f'{response}, error at result {e}\n'
                        self.ana[anno_id][action_uid] = compare_result
                        continue
                    
                else:
                    compare_result['determine_reason'] = self.ana[anno_id][action_uid]['determine_reason']
                    compare_result['determine_result'] = self.ana[anno_id][action_uid]['determine_result']
                    compare_result['error_msg'] = self.ana[anno_id][action_uid]['error_msg']

                if compare_result['determine_result'] == 'same' and click_point_in_bbox:
                    compare_result['determine_reliability'] = 1
                elif compare_result['determine_result'] == 'not the same' and not click_point_in_bbox:
                    compare_result['determine_reliability'] = 1
                else:
                    compare_result['determine_reliability'] = 0

                self.ana[anno_id][action_uid] = compare_result
                    
        u.write_json(self.ana_file, self.ana, encoding='utf-8')

if __name__ == "__main__":
    ma = MMMWAna(sys.argv)
    ma.infer()