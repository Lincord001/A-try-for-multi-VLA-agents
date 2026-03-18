import os
import sys
import time

# ==========================================
# 🔥 模型权重版本配置 - 切换权重时只需取消注释对应版本
# ==========================================
# 版本1: 原始版本
# MODEL_CONFIG = {
#     'model_path': './ckpt/pi0_base/pretrained_model',
#     'dataset_repo_id': 'omy_base_data',
#     'dataset_root': './demo_data_base',
#     'chunk_size': 10,
#     'n_action_steps': 10
# }

# 版本2: ver_2 版本
# MODEL_CONFIG = {
#     'model_path': './ckpt/pi0_base/pretrained_model_ver_2/pretrained_model',
#     'dataset_repo_id': 'omy_base_data_ver_2_clean',
#     'dataset_root': './demo_data_base_ver_2_clean',
#     'chunk_size': 20,
#     'n_action_steps': 20
# }

# 版本3: ver_3 版本（当前使用）
MODEL_CONFIG = {
    'model_path': './ckpt/pi0_base/pretrained_model_ver_3/pretrained_model',
    'dataset_repo_id': 'omy_base_data_ver_3',
    'dataset_root': './demo_data_base_ver_3',
    'chunk_size': 20,
    'n_action_steps': 20
}

# ==========================================
# 🔥 控制频率配置（Hz）- 可根据需要手动修改
# ==========================================
CONTROL_FREQUENCY = 20  # 控制频率，单位：Hz（每秒循环次数）
CONTROL_DT = 1.0 / CONTROL_FREQUENCY  # 控制周期，单位：秒

# 1. 设置 Hugging Face 镜像
print("Setting up environment variables for Hugging Face...")
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HUGGINGFACE_HUB_ENDPOINT'] = 'https://hf-mirror.com'

import threading
import copy
import numpy as np
import torch
from PIL import Image
import torchvision
from torchvision import transforms
import glfw



try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
    from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
    from lerobot.configs.types import FeatureType
    from lerobot.common.datasets.utils import dataset_to_policy_features
    from mujoco_env.y_env3 import SimpleEnv3
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)

# ==========================================
# 📊 性能监控工具
# ==========================================
class PerformanceMonitor:
    """性能监控工具，记录各步骤的执行时间"""
    def __init__(self, window_size=50):
        self.window_size = window_size
        self.main_thread_times = {
            'grab_image': [],
            'data_copy': [],
            'get_action': [],
            'step_env': [],
            'render': [],
            'total_loop': []
        }
        self.inference_thread_times = {
            'lock_read': [],
            'img_resize': [],
            'img_totensor': [],
            'img_todevice': [],
            'normalize_inputs': [],
            'prepare_images': [],
            'prepare_state': [],
            'prepare_language': [],
            'sample_actions': [],
            'postprocess': [],
            'total_inference': []
        }
        self.lock = threading.Lock()
        
    def record_main(self, stage, duration):
        """记录主线程各阶段耗时"""
        with self.lock:
            if stage in self.main_thread_times:
                self.main_thread_times[stage].append(duration)
                if len(self.main_thread_times[stage]) > self.window_size:
                    self.main_thread_times[stage].pop(0)
    
    def record_inference(self, stage, duration):
        """记录推理线程各阶段耗时"""
        with self.lock:
            if stage in self.inference_thread_times:
                self.inference_thread_times[stage].append(duration)
                if len(self.inference_thread_times[stage]) > self.window_size:
                    self.inference_thread_times[stage].pop(0)
    
    def get_stats(self, stage, times_list):
        """获取统计信息"""
        if len(times_list) == 0:
            return 0.0, 0.0, 0.0
        arr = np.array(times_list)
        return np.mean(arr), np.min(arr), np.max(arr)
    
    def print_stats(self, step):
        """打印统计信息（每N步打印一次）"""
        with self.lock:
            print(f"\n{'='*80}")
            print(f"📊 Performance Stats (Step {step})")
            print(f"{'='*80}")
            
            print("\n[主线程] 时间统计 (ms):")
            for stage, times in self.main_thread_times.items():
                mean, min_val, max_val = self.get_stats(stage, times)
                count = len(times)
                if count > 0:
                    print(f"  {stage:15s}: 平均={mean*1000:6.2f} 最小={min_val*1000:6.2f} 最大={max_val*1000:6.2f} (样本={count})")
            
            print("\n[推理线程] 时间统计 (ms):")
            for stage, times in self.inference_thread_times.items():
                mean, min_val, max_val = self.get_stats(stage, times)
                count = len(times)
                if count > 0:
                    print(f"  {stage:15s}: 平均={mean*1000:6.2f} 最小={min_val*1000:6.2f} 最大={max_val*1000:6.2f} (样本={count})")
            
            # 计算推理线程总时间
            total_inference = self.inference_thread_times.get('total_inference', [])
            if len(total_inference) > 0:
                mean_total = np.mean(total_inference)
                print(f"\n🎯 推理线程总耗时: {mean_total*1000:.2f}ms (目标: <250ms)")
            print(f"{'='*80}\n")

# ==========================================
# 🔥 核心修改：异步推理运行器 (The Brain)
# ==========================================
class AsyncInferenceRunner:
    def __init__(self, policy, device, img_transform, control_dt, perf_monitor=None):
        self.policy = policy
        self.device = device
        self.img_transform = img_transform  # 图像预处理函数
        self.control_dt = control_dt  # 控制周期，单位：秒
        self.perf_monitor = perf_monitor  # 性能监控器
        
        # 线程同步锁
        self.lock = threading.Lock()
        
        # 共享数据：输入 (由主线程写入，存储原始数据)
        self.latest_raw_images = None  # 原始图像字典
        self.latest_state = None       # 状态数组
        self.latest_task = None        # 任务字符串列表
        self.latest_obs_timestamp = 0
        
        # 共享数据：输出 (由推理线程写入)
        self.latest_action_chunk = None  # 存储完整的动作块（chunk_size 步动作）
        self.chunk_start_timestamp = 0   # 这组动作对应的观测时间 T0
        
        # 推理频率控制：记录上次处理的观测时间戳，避免重复推理
        self.last_processed_timestamp = 0
        
        # 线程控制
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.thread.start()
        print("🚀 [AsyncRunner] 推理线程已启动")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        print("🛑 [AsyncRunner] 推理线程已停止")

    def update_observation(self, images_dict, state, task, timestamp):
        """
        主线程调用：更新最新的观测数据（原始数据）
        
        Parameters:
            images_dict: 原始图像字典，如 {'front': np.array, 'left': np.array, 'right': np.array}
            state: 状态数组（numpy）
            task: 任务字符串列表，如 ["Go to the workbench."]
            timestamp: 观测时间戳
        """
        t0 = time.perf_counter()
        
        # 🔥 方案一：深度拷贝原始数据，避免数据竞争
        # 主线程可能会继续修改原始数据，导致推理线程读到不一致数据
        images_copy = {}
        for key, value in images_dict.items():
            if isinstance(value, np.ndarray):
                images_copy[key] = value.copy()  # 拷贝 numpy 数组
        
        state_copy = np.array(state, copy=True) if state is not None else None
        task_copy = task.copy() if isinstance(task, list) else task
        
        t1 = time.perf_counter()
        if self.perf_monitor:
            self.perf_monitor.record_main('data_copy', t1 - t0)
        
        with self.lock:
            self.latest_raw_images = images_copy
            self.latest_state = state_copy
            self.latest_task = task_copy
            self.latest_obs_timestamp = timestamp

    def get_action_at_time(self, current_time):
        """
        主线程调用：根据当前时间，从最新的动作块中“切”出正确的一步。
        实现了滑动窗口 + 延迟补偿逻辑。
        """
        with self.lock:
            if self.latest_action_chunk is None:
                return None, "Wait for init"
            
            # 🔥 修复：如果时间戳为0（初始值），说明还没有有效动作块
            if self.chunk_start_timestamp == 0:
                return None, "Wait for init"
            
            chunk = self.latest_action_chunk
            start_time = self.chunk_start_timestamp
            
        # 计算时间差：现在距离观测时刻过去了多久？
        # 例如：观测是在 T=0 拍的，推理花了 0.25s。现在是 T=0.25s。
        # time_delta = 0.25
        time_delta = current_time - start_time
        
        # 计算索引：应该取第几步动作？
        # index = 0.25 / (1/15) = 3.75 ≈ 3 (15Hz)
        step_index = int(time_delta / self.control_dt)
        
        # 边界检查
        chunk_len = chunk.shape[0]  # 动作块的长度（等于 n_action_steps）
        
        if step_index < 0:
            # 理论上不应该发生，除非时钟不同步
            return chunk[0], "Sync Error (Future)"
            
        if step_index < chunk_len:
            # ✅ 正常情况：取对应时间步的动作
            # 比如推理用了5步的时间，我们就直接执行第6步(index=5)
            # 这样就掩盖了推理的延迟
            return chunk[step_index], f"OK (Step {step_index})"
        else:
            # ❌ 超时情况：现在的时刻已经超出了动作块覆盖的未来范围
            # 比如过了 0.6秒，但动作只管 0.5秒
            return None, "Timeout (Stale Chunk)"

    def _inference_loop(self):
        """后台线程：死循环执行推理"""
        while self.running:
            # 1. 获取原始输入数据 (加锁读取)
            t_lock_start = time.perf_counter()
            raw_images = None
            state = None
            task = None
            obs_timestamp = 0
            
            with self.lock:
                t_lock_read = time.perf_counter()
                if self.perf_monitor:
                    self.perf_monitor.record_inference('lock_read', t_lock_read - t_lock_start)
                
                if self.latest_raw_images is not None:
                    # 🔥 修复问题2：只有在新观测到达时才进行推理
                    # 避免对同一观测重复推理（推理速度 > 观测频率的情况）
                    if self.latest_obs_timestamp > self.last_processed_timestamp:
                        raw_images = self.latest_raw_images
                        state = self.latest_state
                        task = self.latest_task
                        obs_timestamp = self.latest_obs_timestamp
                        # 立即标记为已处理，防止在推理过程中对新观测重复推理
                        self.last_processed_timestamp = obs_timestamp
                    else:
                        # 观测没有更新，跳过此次推理
                        raw_images = None
            
            if raw_images is None:
                time.sleep(0.001) # 避免空转占满 CPU
                continue
            
            # 2. 🔥 方案一：在后台线程进行图像预处理（避免主线程占用 GIL/GPU）
            # 这样主线程就不会打断推理线程了
            t_inference_start = time.perf_counter()
            try:
                # 2.1 预处理图像：resize
                t0 = time.perf_counter()
                front_img = Image.fromarray(raw_images['front']).resize((256,256))
                left_img = Image.fromarray(raw_images['left']).resize((256,256))
                right_img = Image.fromarray(raw_images['right']).resize((256,256))
                t1 = time.perf_counter()
                if self.perf_monitor:
                    self.perf_monitor.record_inference('img_resize', t1 - t0)
                
                # 2.2 ToTensor
                t0 = time.perf_counter()
                front_t = self.img_transform(front_img).unsqueeze(0)
                left_t = self.img_transform(left_img).unsqueeze(0)
                right_t = self.img_transform(right_img).unsqueeze(0)
                t1 = time.perf_counter()
                if self.perf_monitor:
                    self.perf_monitor.record_inference('img_totensor', t1 - t0)
                
                # 2.3 .to(device)
                t0 = time.perf_counter()
                front = front_t.to(self.device)
                left = left_t.to(self.device)
                right = right_t.to(self.device)
                state_tensor = torch.tensor(np.array(state, dtype=np.float32)).unsqueeze(0).to(self.device)
                t1 = time.perf_counter()
                if self.perf_monitor:
                    self.perf_monitor.record_inference('img_todevice', t1 - t0)
                
                # 2.4 构建 batch
                batch = {
                    'observation.state': state_tensor,
                    'observation.images.front': front,
                    'observation.images.left': left,
                    'observation.images.right': right,
                    'task': task
                }
                
                # 2.5 执行推理 (这是耗时操作，约 250ms)
                # 🔥 注意：这里不能调用 policy.select_action，因为那个函数有队列陷阱。
                # 我们必须手动调用 modeling_pi0.py 里底层的逻辑。
                with torch.no_grad():
                    # 归一化输入
                    t0 = time.perf_counter()
                    batch = self.policy.normalize_inputs(batch)
                    t1 = time.perf_counter()
                    if self.perf_monitor:
                        self.perf_monitor.record_inference('normalize_inputs', t1 - t0)
                    
                    # 准备数据 (Pi0 特有的预处理)
                    t0 = time.perf_counter()
                    images, img_masks = self.policy.prepare_images(batch)
                    t1 = time.perf_counter()
                    if self.perf_monitor:
                        self.perf_monitor.record_inference('prepare_images', t1 - t0)
                    
                    t0 = time.perf_counter()
                    state_processed = self.policy.prepare_state(batch)
                    t1 = time.perf_counter()
                    if self.perf_monitor:
                        self.perf_monitor.record_inference('prepare_state', t1 - t0)
                    
                    t0 = time.perf_counter()
                    lang_tokens, lang_masks = self.policy.prepare_language(batch)
                    t1 = time.perf_counter()
                    if self.perf_monitor:
                        self.perf_monitor.record_inference('prepare_language', t1 - t0)
                    
                    # 核心推理 (生成动作块，大小为 n_action_steps)
                    t0 = time.perf_counter()
                    actions = self.policy.model.sample_actions(
                        images, img_masks, lang_tokens, lang_masks, state_processed
                    )
                    t1 = time.perf_counter()
                    if self.perf_monitor:
                        self.perf_monitor.record_inference('sample_actions', t1 - t0)
                    
                    # 后处理：去除填充
                    t0 = time.perf_counter()
                    original_action_dim = self.policy.config.action_feature.shape[0]
                    actions = actions[:, :, :original_action_dim]
                    
                    # 反归一化
                    actions = self.policy.unnormalize_outputs({"action": actions})["action"]
                    
                    # (可选) Aloha 编码适配
                    if self.policy.config.adapt_to_pi_aloha:
                        actions = self.policy._pi_aloha_encode_actions(actions)
                    t1 = time.perf_counter()
                    if self.perf_monitor:
                        self.perf_monitor.record_inference('postprocess', t1 - t0)
                
                # 3. 写入输出 (加锁写入)
                # actions shape: (1, n_action_steps, dim) -> 取 batch 0 -> (n_action_steps, dim)
                chunk_np = actions[0].cpu().numpy()
                t_inference_end = time.perf_counter()
                if self.perf_monitor:
                    self.perf_monitor.record_inference('total_inference', t_inference_end - t_inference_start)
                
                with self.lock:
                    self.latest_action_chunk = chunk_np
                    # 关键：我们将这个动作块的"开始时间"标记为观测被捕获的时间
                    # 这样主线程才能知道这个动作块对应哪个时间点
                    self.chunk_start_timestamp = obs_timestamp
                    
            except Exception as e:
                print(f"Inference Error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)

# ==========================================
# 主程序
# ==========================================

DEPLOY_MODE = 'base'

def get_default_transform():
    return transforms.Compose([transforms.ToTensor()])

def main():
    assert DEPLOY_MODE == 'base'
    print(f"🚗 DEPLOYMENT MODE: {DEPLOY_MODE.upper()}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. 加载模型
    print("Loading dataset metadata & Policy...")
    print(f"📦 使用模型配置:")
    print(f"   - 模型路径: {MODEL_CONFIG['model_path']}")
    print(f"   - 数据集: {MODEL_CONFIG['dataset_repo_id']}")
    print(f"   - 数据根目录: {MODEL_CONFIG['dataset_root']}")
    print(f"   - chunk_size: {MODEL_CONFIG['chunk_size']}")
    print(f"   - n_action_steps: {MODEL_CONFIG['n_action_steps']}")
    try:
        dataset_metadata = LeRobotDatasetMetadata(MODEL_CONFIG['dataset_repo_id'], root=MODEL_CONFIG['dataset_root'])
        # ... (配置加载部分保持不变)
        features = dataset_to_policy_features(dataset_metadata.features)
        output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
        input_features = {key: ft for key, ft in features.items() if key not in output_features}
        
        # 创建配置（使用配置字典中的值）
        cfg = PI0Config(input_features=input_features, output_features=output_features, 
                        chunk_size=MODEL_CONFIG['chunk_size'], n_action_steps=MODEL_CONFIG['n_action_steps'])
        
        # 加载模型
        policy = PI0Policy.from_pretrained(MODEL_CONFIG['model_path'], config=cfg, dataset_stats=dataset_metadata.stats)
        
        # 验证关键配置参数是否匹配
        config_mismatch = False
        if policy.config.chunk_size != MODEL_CONFIG['chunk_size']:
            print(f"⚠️  警告: 模型配置 chunk_size={policy.config.chunk_size} 与配置中的 {MODEL_CONFIG['chunk_size']} 不一致")
            config_mismatch = True
        if policy.config.n_action_steps != MODEL_CONFIG['n_action_steps']:
            print(f"⚠️  警告: 模型配置 n_action_steps={policy.config.n_action_steps} 与配置中的 {MODEL_CONFIG['n_action_steps']} 不一致")
            config_mismatch = True
        
        if config_mismatch:
            print("⚠️  配置不匹配可能导致模型行为异常，请检查 MODEL_CONFIG 设置")
        else:
            print("✅ 配置验证通过")
        
        policy.to(device)
        policy.eval()
        print("✅ Policy Loaded.")
    except Exception as e:
        print(f"Initialization Error: {e}")
        return

    # 2. 初始化环境
    print("Initializing MuJoCo...")
    xml_path = './asset/example_scene_y3.xml'
    # 启动时在构造函数内按 DEPLOY_MODE 完成一次 reset，避免重复 reset 拉长初始化。
    PnPEnv = SimpleEnv3(xml_path, action_type='eef_pose', state_type='joint_angle', init_mode=DEPLOY_MODE)

    # 3. 初始化图像预处理
    IMG_TRANSFORM = get_default_transform()
    
    # 4. 初始化性能监控器
    perf_monitor = PerformanceMonitor(window_size=100)
    
    # 5. 初始化异步推理器 (The Runner)
    # 🔥 方案一：将 img_transform 传入，让后台线程进行预处理
    runner = AsyncInferenceRunner(policy, device, IMG_TRANSFORM, control_dt=CONTROL_DT, perf_monitor=perf_monitor)
    control_mode = 'manual'
    step = 0
    
    # 打印操作指南
    print("\n" + "="*60)
    print("🚀 ASYNC MODE READY")
    print("Controls: [N] Auto (Async), [M] Manual, [Q] Quit")
    print("="*60 + "\n")

    try:
        while PnPEnv.env.is_viewer_alive():
            # [A] 物理环境步进 (必须高频运行，不受推理阻碍)
            PnPEnv.step_env()
            
            # [B] 控制循环（频率由 CONTROL_FREQUENCY 控制）
            if PnPEnv.env.loop_every(HZ=CONTROL_FREQUENCY):
                current_time = time.time()
                
                # --- 键位处理 ---
                if PnPEnv.env.is_key_pressed_once(key=glfw.KEY_N):
                    if control_mode == 'manual':
                        control_mode = 'auto'
                        runner.start() # 启动线程
                        print("\n🤖 [AUTO] Async Inference Started!")
                
                if PnPEnv.env.is_key_pressed_once(key=glfw.KEY_M):
                    if control_mode == 'auto':
                        control_mode = 'manual'
                        runner.stop() # 停止线程
                        print("\n👤 [MANUAL] Switched to Manual")

                # --- 控制逻辑 ---
                if control_mode == 'manual':
                    action, reset = PnPEnv.teleop_robot(mode=DEPLOY_MODE)
                    if reset: PnPEnv.reset(mode=DEPLOY_MODE)
                    _ = PnPEnv.step(action, mode=DEPLOY_MODE)
                    PnPEnv.render(teleop=True, idx=step)
                    
                elif control_mode == 'auto':
                    t_loop_start = time.perf_counter()
                    
                    # 1. 收集观测数据（原始数据，不进行预处理）
                    t0 = time.perf_counter()
                    state = PnPEnv.get_base_state()
                    images_dict = PnPEnv.grab_image()  # 获取原始图像，同时更新 self.rgb_* 用于显示
                    t1 = time.perf_counter()
                    if perf_monitor:
                        perf_monitor.record_main('grab_image', t1 - t0)
                    
                    obs_capture_time = time.time() # 关键：记录观测时间点 T_obs
                    
                    # 2. 🔥 方案一：直接将原始数据传给后台线程（非阻塞）
                    # 预处理将在后台线程中进行，避免主线程占用 GIL/GPU
                    runner.update_observation(images_dict, state, [PnPEnv.instruction], obs_capture_time)
                    
                    # 3. 尝试获取当前时刻应该执行的动作 (非阻塞)
                    # runner 会自动计算延迟补偿，返回正确的动作
                    t0 = time.perf_counter()
                    action_step, status_msg = runner.get_action_at_time(time.time())
                    t1 = time.perf_counter()
                    if perf_monitor:
                        perf_monitor.record_main('get_action', t1 - t0)
                    
                    # 4. 执行动作
                    t0 = time.perf_counter()
                    if action_step is not None:
                        # 只取前两维 (Left/Right velocity)
                        action_cmd = action_step[:2]
                        PnPEnv.step(action_cmd, mode=DEPLOY_MODE)
                    else:
                        # 如果没有可用动作 (比如刚启动，或者推理严重超时)
                        # 安全策略：停止
                        PnPEnv.step(np.array([0.0, 0.0]), mode=DEPLOY_MODE)
                    t1 = time.perf_counter()
                    if perf_monitor:
                        perf_monitor.record_main('step_env', t1 - t0)
                    
                    # 5. 渲染和调试信息
                    t0 = time.perf_counter()
                    PnPEnv.render(teleop=False, idx=step)
                    t1 = time.perf_counter()
                    if perf_monitor:
                        perf_monitor.record_main('render', t1 - t0)
                    
                    t_loop_end = time.perf_counter()
                    if perf_monitor:
                        perf_monitor.record_main('total_loop', t_loop_end - t_loop_start)
                    
                    # 6. 打印统计信息
                    if step % 50 == 0 and perf_monitor:  # 每50步打印一次详细统计
                        perf_monitor.print_stats(step)
                    elif step % 10 == 0:
                        # 简单输出
                        current_time_debug = time.time()
                        with runner.lock:
                            chunk_start = runner.chunk_start_timestamp
                        if chunk_start > 0:
                            lag = (current_time_debug - chunk_start) / CONTROL_DT
                        else:
                            lag = float('inf')  # 表示还没有有效动作块
                        print(f"Step {step} | Status: {status_msg} | Lag: {lag:.1f} steps")
                    
                    step += 1

    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        runner.stop()
        if PnPEnv.env.viewer: PnPEnv.env.close_viewer()

if __name__ == "__main__":
    main()