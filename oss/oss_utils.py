import os
import json
import sys
import time
parent_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(root_dir)
from kutils import DEBUG, INFO, WARN, ERROR
import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider

def get_credentials():
    access_key_id = os.getenv('OSS_ACCESS_KEY_ID')
    access_key_secret = os.getenv('OSS_ACCESS_KEY_SECRET')

    if not access_key_id:
        raise ValueError("Access key id should not be null or empty.")
    if not access_key_secret:
        raise ValueError("Secret access key should not be null or empty.")
    return access_key_id, access_key_secret

def oss_upload(ak, sk, oss_filename, content_to_be_upload):
    os.environ['OSS_ACCESS_KEY_ID'], os.environ['OSS_ACCESS_KEY_SECRET'] = ak,sk
    auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())
    bucket = oss2.Bucket(auth, 'https://eos.aliyuncs.com', 'guixgpu-act-demo')
    bucket.put_object(oss_filename, content_to_be_upload)
    url=bucket.sign_url("GET", oss_filename, 300, slash_safe=True)
    return url