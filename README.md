# LeRobot MuJoCo Dual-Mode VLA Tutorial

本工程是一个基于 MuJoCo、LeRobot 和 PI0 策略模型的移动操作仿真项目。项目同时支持机械臂 `arm` 与移动底盘 `base` 两类控制模式，覆盖仿真环境、数据采集、策略部署、RAG 导航、任务拆解、VLM 校验和执行轨迹记录等模块。

代码仓库主要保留工程代码、配置和轻量资源。训练数据、checkpoint、拓扑图输出等大文件可从本文“数据与模型说明”中的网盘链接下载。

## 项目定位

本项目为南方科技大学深港微电子学院 2026 届本科毕业设计相关工程代码。当前版本在原始 LeRobot MuJoCo 教程工程的基础上进行了较大规模扩展，重点从单一桌面抓取示例发展为面向移动操作任务的双智能体/双模式 VLA 仿真、数据采集、部署与任务编排系统。

## 功能概览

- MuJoCo 仿真环境：OMY 机械臂、TurtleBot3 移动底盘、桌面、托盘、杯子等场景资产。
- 双模式数据采集：支持 `arm` / `base` 热切换、手动遥控、专家策略、全自动批量采集。
- PI0 双模式部署：支持 arm/base 模型加载、异步推理、动作平滑、base 后处理和任务循环统计。
- Policy Server：可将模型推理常驻在单独进程中，部署脚本通过本地连接请求推理。
- Embodied-RAG：支持从 LeRobot 数据集构建拓扑图、语义图、森林图，并进行自然语言检索和稠密路径规划。
- 任务编排：支持自然语言任务拆解、base/arm 串行任务队列、VLM 执行校验和 replanning 扩展。
- 数据辅助工具：包含数据集清洗、可视化、episode 修复、输入尺寸校验、权重合并等脚本。

## 环境要求

建议在 Linux + NVIDIA GPU 环境下运行。项目默认使用 CUDA 12.4 对应的 PyTorch 版本。

所有会导入项目模块、运行 Python 校验、执行脚本或启动工具的命令，都需要先激活 Conda 环境：

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate vla_fresh
```

不要依赖 `~/.bashrc` 在非交互 shell 中自动激活环境。

安装依赖：

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate vla_fresh
pip install -r requirements.txt
```

`requirements.txt` 中主要依赖包括：

- `mujoco==3.1.6`
- `torch==2.6.0` / `torchvision==0.21.0` / `torchaudio==2.6.0`
- `transformers==4.50.3`
- `datasets==3.4.1`
- `safetensors==0.5.3`
- `lerobot` 指定 commit 版本

如果使用 RAG、任务拆解或 VLM 校验中调用 DashScope/Qwen 的功能，需要配置：

```bash
export DASHSCOPE_API_KEY="你的 DashScope API Key"
```

## 目录结构

```text
.
├── 7_2_pi0_deploy_v7_dual_8_hybrid.py   # 双模式 PI0/VLA 部署主入口
├── collect_data_v7.py                    # arm/base 数据采集主入口
├── record_execution_tracker_trace.py     # ExecutionTracker 输入/输出 JSONL 记录器
├── requirements.txt
├── asset/                                # MuJoCo XML、机器人、桌面和物体资产
├── ckpt/                                 # 本地策略模型 checkpoint
├── collect_data_v7_modules/              # 数据采集配置、状态机、保存线程、按键处理
├── deploy_v7_dual/                       # 部署配置、模型加载、异步推理、控制循环、任务队列
├── demo_data_arm_* / demo_data_base_*    # LeRobot 格式采集数据集
├── mujoco_env/                           # MuJoCo 环境、IK、专家策略、渲染、遥控、动作工具
├── orchestration/                        # 任务拆解、replanner、VLM verifier、execution tracker
├── rag_pipeline/                         # Embodied-RAG 建图、语义、森林、检索、路径规划
├── topology_output/                      # 拓扑图、语义图、森林图、embedding cache
├── assit files/                          # 数据集检查、清洗、可视化等辅助脚本
└── tests/                                # 当前测试用例
```

## 快速开始

### 1. 检查 Python 文件能否编译

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate vla_fresh
python -m compileall collect_data_v7_modules deploy_v7_dual mujoco_env orchestration rag_pipeline
```

### 2. 运行测试

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate vla_fresh
pytest tests
```

当前测试主要覆盖任务拆解器的离线逻辑。

## 数据采集

数据采集入口：

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate vla_fresh
python collect_data_v7.py
```

常用配置位于：

```text
collect_data_v7_modules/config.py
```

重点配置项：

- `GPU_DEVICE_ID`：选择使用的 GPU。
- `INITIAL_MODE`：启动模式，支持 `arm` 或 `base`。
- `ARM_DATASET_ROOT` / `BASE_DATASET_ROOT`：采集数据保存位置。
- `IMG_SIZE` / `FPS` / `MAX_EPISODE_SEC`：图像尺寸、采集频率和单条 episode 时长上限。
- `RANDOM_INIT_ENABLED`：随机初始化模式，`0` 关闭，`1` 扇形区域，`2` 圆形交集。
- `MULTI_CONFIG_RECORDING`：多阶段全自动采集配置。

采集脚本启动后主要按键：

| 按键 | 功能 |
| --- | --- |
| `J` | 开始录制 |
| `K` | 停止并保存 |
| `I` | 丢弃当前录制 |
| `Z` | 仅重置环境 |
| `C` | 在 `base` / `arm` 模式间热切换 |
| `←` / `→` | 切换当前模式的 instruction group |
| `H` | base 模式自动停车并保存 |
| `T` | arm 模式专家策略测试执行，不录制 |
| `Y` | arm 模式专家策略执行 + 录制 + 自动保存 |
| `O` | arm 平滑回 home |
| `P` | 开启/停止全自动批量录制 |

采集数据使用 LeRobot 格式保存。当前默认测试路径包括：

```text
demo_data_arm_test/
demo_data_base_test/
```

训练/部署用数据路径在部署配置中还包括：

```text
demo_data_arm_v7_1/
demo_data_base_v7_6/
```

## 模型训练

训练入口：

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate vla_fresh
python train_model.py --config_path=pi0_arm_v7_1.yaml
```

Base 模型训练：

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate vla_fresh
python train_model.py --config_path=pi0_base_v7_6.yaml
```

两份训练配置分别位于：

```text
pi0_arm_v7_1.yaml
pi0_base_v7_6.yaml
```

默认训练配置使用的数据集和输出路径为：

```text
demo_data_arm_v7_1/                  # arm 7.1 训练数据集
demo_data_base_v7_6/                 # base 7.6 训练数据集
ckpt/pi0_arm/pretrained_model_arm_v7_1/
ckpt/pi0_base/pretrained_model_base_v7_6/
```

训练所需的大文件，包括 `asset/` 场景资产、`demo_data_arm_v7_1/`、`demo_data_base_v7_6/`、已训练好的 arm/base checkpoint，以及训练 PI0 所需的 PaliGemma 权重，均可从“数据与模型说明”中的百度网盘链接下载。

如果本机已经有 PaliGemma 权重，可以通过环境变量指定本地路径：

```bash
export PALIGEMMA_PRETRAINED_PATH="./PaliGemmaWeights/paligemma-3b-pt-224"
```

如果不设置该环境变量，`train_model.py` 默认使用 Hugging Face 模型 ID：

```text
google/paligemma-3b-pt-224
```

训练配置中还保留了几项已注释的 checkpoint 保存字段：

```text
save_by_freq
save_steps
save_by_steps
best_loss_checkpoint_range
save_best_in_range
```

这些是作者训练时为了节省磁盘空间、在指定 step 保存 checkpoint、或只保留某个区间内最低 loss checkpoint 使用的小工程 trick。不同 LeRobot 版本的 `TrainPipelineConfig` 不一定支持这些字段；如果训练启动时报 `not valid for TrainPipelineConfig`，保持这些字段注释即可使用标准 LeRobot 保存逻辑。

## 双模式部署

部署入口：

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate vla_fresh
python 7_2_pi0_deploy_v7_dual_8_hybrid.py
```

部署配置位于：

```text
deploy_v7_dual/config.py
```

重点配置项：

- `ARM_CONFIG`：arm 模型路径、数据集、图像尺寸、状态维度、动作维度、相机键。
- `BASE_CONFIG`：base 模型路径、数据集、图像尺寸、状态维度、动作维度、相机键。
- `LOAD_ARM_MODEL` / `LOAD_BASE_MODEL`：是否加载对应模型。
- `POLICY_SERVER_ENABLED`：是否使用本地常驻推理服务。
- `ARM_INFERENCE_MODE`：`sync` 或 `async`。
- `ACTION_HORIZON` / `BASE_ACTION_HORIZON`：异步动作 chunk 执行长度。
- `SMOOTHING_ENABLED`：arm 动作平滑。
- `BASE_POSTPROC_ENABLED`：base 动作后处理。
- `TASK_TIMEOUT_SEC` / `TASK_LOOP_COUNT`：任务超时和循环次数。

部署脚本主要按键：

| 按键 | 功能 |
| --- | --- |
| `C` | 切换 `ARM` / `BASE` 模式，不重置环境 |
| `N` | 启动当前模式 PI0 自动控制 |
| `M` | 切回人工遥控 |
| `Z` | 重置环境 |
| `K` | 切换 base 速度缩放 |
| `←` / `→` | 切换当前模式 instruction group |
| `L` | arm 自动控制 + 自动检测成功/失败 |
| `G` | 将 base 传送到预设位置 |
| `T` | base 模式自然语言检索导航目标 |
| `U` | arm 模式自然语言检索标准化指令 |
| `Y` | 拆解自然语言任务并启动串行任务队列 |
| `R` | 切换 base RAG Navigation |
| `P` | 切换 JSON 串行任务队列 |
| `Q` | 退出 |

### 使用 Policy Server

如果 `deploy_v7_dual/config.py` 中 `POLICY_SERVER_ENABLED = True`，需要先启动本地推理服务：

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate vla_fresh
python -m deploy_v7_dual.policy_server
```

默认监听：

```text
127.0.0.1:59600
```

然后再另开终端启动部署脚本。

## Embodied-RAG 流程

统一入口：

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate vla_fresh
python -m rag_pipeline.rag_main --help
```

RAG 分为五个阶段：

1. Stage 1：从 LeRobot 数据集构建拓扑图。
2. Stage 2：为拓扑节点生成语义描述。
3. Stage 3：聚类并构建语义森林。
4. Stage 4：根据自然语言 query 检索目标节点。
5. Stage 5：根据当前位姿和目标节点规划稠密路径点。

### 构建拓扑/语义/森林图

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate vla_fresh
export DASHSCOPE_API_KEY="你的 DashScope API Key"
python -m rag_pipeline.rag_main build \
  --dataset_dir ./demo_data_base_RAG \
  --output_dir topology_output
```

默认输出：

```text
topology_output/topological_map.json
topology_output/semantic_topological_map_text_only.json
topology_output/forest_topological_map.json
topology_output/topological_map.png
topology_output/forest_topological_map.png
```

### 查询目标并规划路径

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate vla_fresh
export DASHSCOPE_API_KEY="你的 DashScope API Key"
python -m rag_pipeline.rag_main query \
  --query "go to the workbench" \
  --start_pose "0.0,0.0,0.0" \
  --forest_json topology_output/forest_topological_map.json \
  --topology_json topology_output/topological_map.json \
  --dense_output_json dense_waypoints_output.json
```

如果已经知道目标节点，可以用 `--target_node` 跳过自然语言检索：

```bash
python -m rag_pipeline.rag_main query \
  --target_node ep2_frame256 \
  --start_pose "0.0,0.0,0.0"
```

## 任务编排与执行记录

任务编排模块位于：

```text
orchestration/
```

主要模块：

- `task_decomposer.py`：将自然语言用户任务拆成可执行的 base/arm 串行任务。
- `task_replanner.py`：任务失败后的重新规划后端。
- `vlm_verifier.py`：基于视觉模型的执行结果校验。
- `execution_tracker.py`：跟踪 base/arm 执行进度和状态。

部署时相关开关在 `deploy_v7_dual/config.py` 中：

- `TASK_DECOMPOSER_ENABLED`
- `TASK_REPLANNER_ENABLED`
- `ARM_VLM_ORCHESTRATION_ENABLED`
- `EXECUTION_TRACE_ENABLED`
- `EXECUTION_TRACE_OUTPUT_DIR`

`record_execution_tracker_trace.py` 可独立记录 tracker 的输入、输出和环境时间戳，便于离线复盘。查看命令行参数：

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate vla_fresh
python record_execution_tracker_trace.py --help
```

## 数据与模型说明

训练、部署和 RAG 相关的大文件已单独打包到网盘，包含训练数据、场景资产、checkpoint、拓扑图输出等运行所需资源：

```text
百度网盘：https://pan.baidu.com/s/1moSiJeqfHYVTdpFbvmmxGw?pwd=1234
提取码：1234
```

下载后请将对应目录放回项目根目录，并确认以下路径与配置文件一致。

项目中常用的大文件目录包括：

```text
ckpt/pi0_arm/
ckpt/pi0_base/
demo_data_arm_v7_1/
demo_data_base_v7_6/
demo_data_base_RAG/
topology_output/
execution_tracker_traces/
vlm_verifier_checks/
```

这些目录通常体积较大。迁移工程时需要确认模型路径、数据路径和 `deploy_v7_dual/config.py` / `collect_data_v7_modules/config.py` 中的路径一致。

## 辅助脚本

辅助脚本位于 `assit files/`。目录名中包含空格，命令行运行时需要加引号：

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate vla_fresh
python "assit files/visualize_dataset.py"
```

常见脚本用途：

- `visualize_dataset.py`：可视化 LeRobot 数据集。
- `view_state_action_data.py`：查看状态和动作数据。
- `view_picture_in_dataset_v4.py`：查看数据集中的图像。
- `clean_dataset.py`：清洗数据集。
- `fix_episode_tasks.py`：修复 episode task 字段。
- `verify_output_size.py` / `verify_pi0_inputs.py`：检查输出和 PI0 输入尺寸。
- `merge_weights.py`：合并权重。
- `analyze_cup_positions.py`：分析杯子位置分布。

## 常见问题

### Import "glfw" could not be solved

这是已知编辑器提示，可以忽略。实际运行以 Conda 环境中的依赖为准。

### 找不到模型或数据集

检查以下配置是否与本机路径一致：

```text
deploy_v7_dual/config.py
collect_data_v7_modules/config.py
```

重点看：

- `ARM_CONFIG["model_path"]`
- `BASE_CONFIG["model_path"]`
- `ARM_CONFIG["dataset_root"]`
- `BASE_CONFIG["dataset_root"]`
- `ARM_DATASET_ROOT`
- `BASE_DATASET_ROOT`

### RAG 或任务拆解报 API Key 错误

需要先设置：

```bash
export DASHSCOPE_API_KEY="你的 DashScope API Key"
```

### 启动部署脚本后连接不上策略模型

如果 `POLICY_SERVER_ENABLED = True`，先确认 `python -m deploy_v7_dual.policy_server` 已启动，并且 host、port、authkey 与 `deploy_v7_dual/config.py` 一致。

## 项目来源与致谢

本仓库最初基于 JeongEun Park / Jeongeun 的开源项目 **LeRobot Tutorial with MuJoCo** 进行开发。原项目提供了 MuJoCo + LeRobot 环境中的数据采集、ACT/PI0/SmolVLA 训练与部署教程，是本毕业设计工程的基础。

在此基础上，本工程进一步扩展了：

- arm/base 双模式控制与热切换；
- TurtleBot3 移动底盘与 OMY 机械臂协同任务；
- PI0 双模型部署、异步推理与本地 policy server；
- 多阶段自动数据采集与 LeRobot 数据集管理；
- Embodied-RAG 拓扑记忆、语义森林、自然语言目标检索与路径规划；
- 自然语言任务拆解、串行任务队列、VLM verifier 和 execution tracker；
- 面向毕业设计实验流程的数据清洗、可视化和分析辅助工具。

同时感谢原始工程 README 中列出的相关开源资源：

- ROBOTIS OMY 机械臂资产来自 `robotis_mujoco_menagerie`。
- `mujoco_env/mujoco_parser.py` 修改自 `yet-another-mujoco-tutorial-v3`。
- 项目参考了 Hugging Face LeRobot examples。
- plate 和 mug 等物体资产来自 Objaverse。

## 开发约定

- 运行任何 Python 脚本前都先激活 `vla_fresh`。
- 修改采集参数优先看 `collect_data_v7_modules/config.py`。
- 修改部署参数优先看 `deploy_v7_dual/config.py`。
- 修改任务文本或 instruction group 优先看 `mujoco_env/instruction_utils.py`。
- 大文件目录如 `ckpt/`、`demo_data_*`、`execution_tracker_traces/`、`vlm_verifier_checks/` 不建议无意中整体提交或复制。
