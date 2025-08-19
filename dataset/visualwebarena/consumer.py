import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
up_dir = parent_dir
for i in range(3):
    sys.path.append(up_dir)
    up_dir = os.path.dirname(up_dir)
from kutils import DEBUG, INFO, WARN, ERROR
import utils as u

import threading
import queue
import time
from typing import List, Any

class TaskManager:
    def __init__(self):
        self.task_queue = queue.Queue()
        self.running = True
        self.worker_thread = None
        self._start_worker()

    def _start_worker(self):
        """启动后台任务处理线程"""
        self.worker_thread = threading.Thread(target=self._function_B, name="TaskWorker", daemon=True)
        self.worker_thread.start()
        INFO("后台任务处理器已启动")

    def _function_B(self):
        """后台任务处理函数：顺序执行队列中的任务"""
        INFO("B 等待任务...")
        while self.running:
            try:
                task = self.task_queue.get(timeout=1)
                if task is None:
                    continue
                self._execute_task(task)
                self.task_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                ERROR(f"B 处理任务时出错: {e}")
                self.task_queue.task_done()

    def _execute_task(self, task_list: List[Any]):
        """执行单个任务"""
        INFO(f"开始执行任务: {task_list}")
        for item in task_list:
            INFO(f"处理项: {item}")
            time.sleep(0.5)  # 模拟处理耗时
        INFO(f"任务完成: {task_list}")

    def function_A(self, task_list: List[Any]):
        """
        A 函数：提交任务给 B（不阻塞）
        """
        self.task_queue.put(task_list)
        INFO(f"A 提交任务: {task_list}")

    def submit_task(self, task_list: List[Any]):
        self.function_A(task_list)

    def wait_until_all_tasks_done(self):
        """
        阻塞主线程，直到所有任务都被 B 处理完成
        """
        INFO("等待所有任务处理完成...")
        self.task_queue.join()  # 精确等待所有 task_done()
        INFO("所有任务已处理完毕")

    def stop(self):
        self.running = False


if __name__ == "__main__":
    tm = TaskManager()

    # === 模拟 function A 提交任务 ===
    tasks = [
        ["文件1.txt", "上传"],
        ["清理缓存", "压缩日志", "备份"],
        ["发送邮件", "收件人A", "收件人B"],
        ["单步任务"]
    ]

    INFO("开始提交任务...")
    for task in tasks:
        tm.function_A(task)
        time.sleep(0.05)  # 模拟 A 边工作边提交

    INFO("A：所有任务提交完毕")

    # === 主线程等待 B 处理完所有任务 ===
    tm.wait_until_all_tasks_done()

    INFO("主线程确认：A 和 B 都已完成，程序安全退出")
