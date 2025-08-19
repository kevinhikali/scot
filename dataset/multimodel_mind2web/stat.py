import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
up_dir = parent_dir
for i in range(3):
    sys.path.append(up_dir)
    up_dir = os.path.dirname(up_dir)
from kutils import DEBUG, INFO, WARN, ERROR
import utils as u
import re
from tqdm import main, tqdm
import matplotlib.pyplot as plt
from dataset_utils import draw_eval

def insert_newlines(s, n):
    return '\n'.join(s[i:i+n] for i in range(0, len(s), n))

def plot_pie(data, fig_name, save_path):
    fig, ax = plt.subplots(figsize=(3, 3), dpi=200)
    fontsize = 12
    ax.set_title(fig_name, fontsize=fontsize)
    xs = list(data.values())
    ys = list(data.keys())
    # colors = ['#d5695d', '#5d8ca8', '#65a479', '#a564c9']
    ax.pie(
        xs, 
        labels=ys, 
        autopct='%1.1f%%', 
        textprops={'fontsize': 8, 'color': 'black'},
        wedgeprops={'edgecolor': 'black', 'linewidth': .5}
    )
    bw = 1
    ax.spines['bottom'].set_linewidth(bw)
    ax.spines['left'].set_linewidth(bw)
    ax.spines['top'].set_linewidth(bw)
    ax.spines['right'].set_linewidth(bw)
    # plt.legend()
    save_file = f'{save_path}/{fig_name}.png'
    plt.savefig(save_file)
    plt.close(fig)

if __name__ == "__main__":
    dataset_path = f'{u.get_nas()}/gui_dataset/MultiModel_Mind2Web/'
    ana_path = f'{dataset_path}/ana/'
    ana_folder = 'SCOT_bbox_sw0_4o_d250624_b5e-2_sft_160_grpo_200_vllm_gpt-4o'
    main_path = f'{ana_path}/{ana_folder}/'
    splits = ['cross_domain', 'cross_task', 'cross_website']

    gt_img_path = f'{dataset_path}/gt_imgs/'
    result_img_path = f'{main_path}/result_imgs/'
    u.mkdir(result_img_path)
    yn_path = f'{result_img_path}/yn/'
    u.mkdir(yn_path)
    ny_path = f'{result_img_path}/ny/'
    u.mkdir(ny_path)
    nn_path = f'{result_img_path}/nn/'
    u.mkdir(nn_path)

    n_steps = 0
    desc_same_y_success_y = []
    desc_same_y_success_n = []
    desc_same_n_success_y = []
    desc_same_n_success_n = []
    gt_error = []
    for split in splits:
        split_path = main_path + '/' + split
        files = u.list_files(split_path, True)
        pattern_train = re.compile(r"^ana_\d+_\d+\.json$")
        files = [a for a in files if pattern_train.match(u.get_name(a))]
        anas = {}
        for file in files: 
            anas.update(u.read_json(file))

        for anno_id in tqdm(anas.keys(), split):
            for action_uid in anas[anno_id].keys():
                gt_img_file = f'{gt_img_path}/{split}/{anno_id}_{action_uid}.png'
                n_steps += 1
                step_content = anas[anno_id][action_uid]
                task = step_content['task']
                pred_click_point = step_content["pred_click_point"]
                gpt_action = step_content["gt_action"]
                if '0, 0, 0, 0' in gpt_action:
                    gt_error.append((anno_id, action_uid))
                    continue
                pred_action = step_content["pred_action"]
                gpt_action_desc = step_content["gpt_action_desc"]
                pred_action_desc = step_content["pred_action_desc"]
                action_type_the_same = step_content["action_type_the_same"]
                type_value_the_same = step_content["type_value_the_same"]
                empty_gt_bbox = step_content["empty_gt_bbox"]
                pred_format_error = step_content["pred_format_error"]
                click_point_in_bbox = step_content["click_point_in_bbox"]
                determine_reason = step_content["determine_reason"]
                determine_reliability = step_content["determine_reliability"]
                determine_result = step_content["determine_result"]
                error_msg = step_content["error_msg"]
                # success_flag = action_type_the_same and type_value_the_same and click_point_in_bbox
                success_flag = click_point_in_bbox
                eval_img_file = ''
                if determine_result == 'same':
                    if success_flag:
                        desc_same_y_success_y.append((anno_id, action_uid))
                    else:
                        desc_same_y_success_n.append((anno_id, action_uid))
                        eval_img_file = f'{yn_path}/{anno_id}_{action_uid}.png'
                elif determine_result == 'not the same':
                    if success_flag:
                        desc_same_n_success_y.append((anno_id, action_uid))
                        eval_img_file = f'{ny_path}/{anno_id}_{action_uid}.png'
                    else:
                        desc_same_n_success_n.append((anno_id, action_uid))
                        eval_img_file = f'{nn_path}/{anno_id}_{action_uid}.png'
                if eval_img_file:
                    if u.is_file_exist(eval_img_file): continue
                    n_insert = 140
                    draw_eval(gt_img_file, pred_click_point, insert_newlines(task, n_insert), 
                              gpt_action, pred_action, 
                              insert_newlines(gpt_action_desc, n_insert), insert_newlines(pred_action_desc, n_insert), 
                              eval_img_file)
    n_desc_same_y_success_y = len(desc_same_y_success_y)
    n_desc_same_y_success_n = len(desc_same_y_success_n)
    n_desc_same_n_success_y = len(desc_same_n_success_y)
    n_desc_same_n_success_n = len(desc_same_n_success_n)
    n_gt_error = len(gt_error)

    pie_data = {
        'yy': n_desc_same_y_success_y,
        'yn': n_desc_same_y_success_n,
        'ny': n_desc_same_n_success_y,
        'nn': n_desc_same_n_success_n,
        # 'ge': n_gt_error,
    }
    plot_pie(pie_data, 'desc_same_point_in_bbox', main_path)
