import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
up_dir = parent_dir
for i in range(3):
    sys.path.append(up_dir)
    up_dir = os.path.dirname(up_dir)
from kutils import DEBUG, INFO, WARN, ERROR
import utils as u

class OSSUtil():
    def __init__(self, bucket, silent = False):
      self.exe = u.get_home() + '/ossutilmac64'
      self.bucket = bucket
      self.silent = silent

    def stat(self):
        command = f"{self.exe} stat oss://{self.bucket}/"
        u.execute(command, self.silent)

    def mkdir(self, oss_rel_path):
        path = f'{self.bucket}/{oss_rel_path}'
        path = path.replace('//', '/')
        command = f'{self.exe} mkdir oss://{path}'
        u.execute(command, self.silent)

    def upload(self, local_file, oss_rel_path):
        path = f'{self.bucket}/{oss_rel_path}'
        path = path.replace('//', '/')
        command = f'{self.exe} cp {local_file} oss://{path}'
        u.execute(command, self.silent)

    def cat(self, oss_rel_file):
        path = f'{self.bucket}/{oss_rel_file}'
        path = path.replace('//', '/')
        command = f'{self.exe} cat oss://{path}'
        u.execute(command, self.silent)

if __name__ == "__main__":
    bucket = 'antsys-aworldspace-prod'
    ou = OSSUtil(bucket)
    ou.stat()
    oss_rel_path = '/ml001/browser_agent/vwa/kevin/'
    ou.mkdir(oss_rel_path)
    local_file = u.get_nas() + '/gui_dataset/visualwebarena/auth/reddit_state.json'
    filename = u.get_name(local_file)
    ou.upload(local_file, oss_rel_path)
    ou.cat(f'{oss_rel_path}/{filename}')

