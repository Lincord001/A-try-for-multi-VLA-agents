"""
data_saver.py
-------------
DataSaverWorker: 支持双模式的异步数据保存工作线程。
图像 resize 在后台线程中完成，主线程不阻塞。
"""

import threading
import queue
import numpy as np
from PIL import Image

from .config import IMG_SIZE


# ================= 🧵 异步处理工作线程 🧵 =================

class DataSaverWorker(threading.Thread):
    """支持双模式的数据保存工作线程 (优化版：无阻塞 + 快速 resize)"""
    def __init__(self, datasets):
        """
        Parameters:
            datasets: dict, {'arm': LeRobotDataset, 'base': LeRobotDataset}
        """
        super().__init__()
        self.datasets = datasets
        # 🔥 关键改动 1：去掉 maxsize 限制，永不阻塞主线程
        self.queue = queue.Queue(maxsize=0)  # 0 = 无限大小
        self.daemon = True
        self.running = True
        self._peak_qsize = 0  # 记录峰值，用于调试
        self.saving_in_progress = False  # 🔥 保存进行中标志（防止竞态条件）

    def put(self, item):
        """主线程调用：将原始数据放入队列（永不阻塞）"""
        self.queue.put_nowait(item)  # 非阻塞 put
        # 更新峰值记录
        current_size = self.queue.qsize()
        if current_size > self._peak_qsize:
            self._peak_qsize = current_size

    def qsize(self):
        """返回当前队列大小"""
        return self.queue.qsize()

    def peak_qsize(self):
        """返回队列峰值大小"""
        return self._peak_qsize

    @staticmethod
    def _fast_resize(img):
        """🔥 V4.1: 改回 PIL resize，与部署环境保持一致"""
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        return np.array(pil_img)

    def run(self):
        """子线程循环：后台处理数据"""
        while self.running or not self.queue.empty():
            try:
                item = self.queue.get(timeout=1.0)
            except queue.Empty:
                # 收到停止信号且队列已空时，安全退出线程
                if not self.running:
                    break
                continue

            mode = item[0]
            images_dict = item[1]
            state = item[2]
            action = item[3]
            obj_init = item[4]
            task = item[5]
            base_pose = item[6] if len(item) > 6 else None

            dataset = self.datasets.get(mode)
            if dataset is None:
                print(f"Warning: No dataset for mode '{mode}'")
                self.queue.task_done()
                continue

            frame_data = {
                "observation.state": state,
                "action": action,
                "obj_init": obj_init,
            }

            try:
                # 🔥 使用快速 resize
                if mode == 'arm':
                    frame_data["observation.images.agent"] = self._fast_resize(images_dict['agent'])
                    frame_data["observation.images.wrist"] = self._fast_resize(images_dict['wrist'])
                elif mode == 'base':
                    frame_data["observation.images.front"] = self._fast_resize(images_dict['front'])
                    frame_data["observation.images.left"] = self._fast_resize(images_dict['left'])
                    frame_data["observation.images.right"] = self._fast_resize(images_dict['right'])
                    if base_pose is not None:
                        frame_data["base_pose"] = base_pose

                # 写入硬盘
                dataset.add_frame(frame_data, task=task)
            except Exception as e:
                print(f"Error in worker thread: {e}")
            finally:
                self.queue.task_done()

    def wait_queue_empty(self):
        """等待所有数据处理完毕（录制结束后调用）"""
        self.queue.join()

    def clear_queue(self):
        """🔥 清空队列中所有未处理的数据（丢弃时调用）"""
        # 🔥 防止在保存期间清空队列（竞态条件保护）
        if self.saving_in_progress:
            print(f"   ⚠️ Cannot clear queue: Save operation in progress. Please wait...")
            return False

        cleared_count = 0
        while True:
            try:
                self.queue.get_nowait()
                self.queue.task_done()
                cleared_count += 1
            except queue.Empty:
                break
        if cleared_count > 0:
            print(f"   🗑️ Cleared {cleared_count} frames from queue.")
        return True
