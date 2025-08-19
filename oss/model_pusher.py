
import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
up_dir = parent_dir
for i in range(3):
    sys.path.append(up_dir)
    up_dir = os.path.dirname(up_dir)
from kutils import DEBUG, INFO, WARN, ERROR
import utils as u

from aistudio_common.utils import model_utils

class ModelPusher():
    def __init__(self, model_path, model_name):
        self.model_path = model_path
        self.parent_path = os.path.dirname(self.model_path)
        self.model_name = model_name

    def push(self):
        pass

class ModelPusherOSS(ModelPusher):
    def __init__(self, model_path, model_name, oss_folder):
        super().__init__(model_path, model_name)
        self.oss_exe = 'ossutil64'
        self.bucket_name = 'antsys-code-agent-bucket'
        self.bucket = f'oss://{self.bucket_name}/'
        self.oss_folder = oss_folder

    def push(self):
        command = f"{self.oss_exe} stat {self.bucket}"
        result = u.execute(command)

        command = f'{self.oss_exe} cp -r {self.model_path} {self.bucket}{self.oss_folder}/'
        result = u.execute(command)
        if result == None: exit()

    def ls(self):
        command = f"{self.oss_exe} ls {self.bucket}"
        result = u.execute(command)
        
class ModelPusherAIS(ModelPusher):
    def __init__(self, model_path, model_name, compression_overwrite=True):
        super().__init__(model_path, model_name)
        self.compression_file = f'{self.parent_path}/{self.model_name}.tar.gz'
        self.compression_overwrite = compression_overwrite

    def compression(self):
        if self.compression_overwrite:
            u.compress_folder_to_tar_gz(self.model_path, self.compression_file)
            return

        if not u.is_file_exist(self.compression_file):
            u.compress_folder_to_tar_gz(self.model_path, self.compression_file)
        return

    def push(self):
        self.compression()
        # 这个代码可选，如果想生成永久访问的 url 链接，请加上环境变量这段代码
        # os.environ["AISTUDIO_SITE_ENUM"] = "INTERNAL"
        oss_path = model_utils.upload_model_to_ais(
            local_path=self.compression_file,
            dst_store_key=f"hekaiwen/{self.model_name}.tar.gz", 
            enable_overwrite=True,
            show_progress=True
            )

        u.write_txt(f'{self.parent_path}/{self.model_name}_oss_path.txt', oss_path)

if __name__ == "__main__":
    model_path = '/ossfs/workspace/D-RPA/llama_fac/saves_qwen_72b_instruct/checkpoint-120'
    # oss_folder = 'model'
    # mp = ModelPusherOSS(model_path, oss_folder)
    # mp.push()
    # mp.ls()

    model_name = 'qwen25_72b_instruct_tool_call_dpo_241028'
    mp = ModelPusherAIS(model_path, model_name, False)
    mp.push()
