# coding: utf-8
# Copyright (c) Antfin, Inc. All rights reserved.

import logging
import logging.handlers

# 设置日志器的配置信息
# LOG_FILENAME = 'my_app.log'
LOGGER_NAME = 'my_app_logger'

# 获取或创建一个logger实例
logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(logging.DEBUG)

# 检查是否已经存在Handler，避免重复创建
if not logger.handlers:
    # 创建一个到标准控制台的Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 如果需要，可以再添加其他Handler，例如一个文件Handler
    # file_handler = logging.FileHandler(LOG_FILENAME)
    # file_handler.setLevel(logging.INFO)
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)

# 设置logconfig.py的Logger只处理WARNING以上级别的日志，
# 避免在配置日志时记录不必要的DEBUG或INFO日志
logging.getLogger(LOGGER_NAME).setLevel(logging.WARNING)
