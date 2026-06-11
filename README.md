# 绝影 Lite3 智能控制系统 🐕

基于**云深处绝影 Lite3** 四足机器人的智能控制系统，集成**人脸识别、情绪识别、手势控制**于一体。

## 📋 功能概览

### 1. 人脸识别与情绪识别
- 使用 **InsightFace** (buffalo_m) 进行实时人脸检测与身份识别
- 使用 **VGG19** 模型对 7 种情绪进行分类：`Angry`、`Disgust`、`Fear`、`Happy`、`Sad`、`Surprise`、`Neutral`
- 5 秒情绪采样窗口，取最 dominant 的情绪触发动作
- 支持人脸数据库录入与管理（`s` 键录入，`d` 键删除）

### 2. 情绪驱动动作序列
| 情绪 | 动作序列 | 安全距离 |
|------|---------|---------|
| Sad (悲伤) | 旋转 → 跳跃 → 后空翻 | 1.0m |
| Happy (高兴) | 太空步 → 扭身 → 挥手 | 0.6m |
| Surprise (惊讶) | 向前跳 → 来回奔跑 | 0.8m |
| Fear (恐惧) | 跳跃 → 匍匐 → 抓地 | 0.3m |
| Angry (生气) | 同 Fear 动作 | 0.3m |
| Disgust (厌恶) | 同 Fear 动作 | 0.3m |
| Neutral (中性) | 无动作 | - |

### 3. 手势实时控制
| 手势 | 动作 | 类型 |
|------|------|------|
| 👊 握拳 | 待命 | 单次 |
| ✋ 五指张开 | 急停 | 单次 |
| 👍 竖拇指 | 起立/趴下 | 单次 |
| 1️⃣ 食指 | 前进 | 持续 |
| ✌️ 剪刀手 | 后退 | 持续 |
| 🤟 三指 | 左转 | 持续 |
| 🖖 四指 | 右转 | 持续 |
| 🤙 六(拇+小) | 中速 | 单次 |
| 👌 OK | 回零 | 单次 |
| ⬅️ 左滑 | 左平移 | 持续 |
| ➡️ 右滑 | 右平移 | 持续 |

### 4. 深度相机安全检测
- 初始启动时 360° 环绕扫查
- 执行动作前检测前方安全距离
- 不同情绪有不同的安全距离阈值

### 5. 双阶段工作流
1. **情绪识别阶段** — 检测人脸 → 收集 5 秒情绪 → 执行对应动作
2. **手势控制阶段** — 动作完成后自动切换，通过手势控制机器狗运动

## 🚀 快速开始

### 环境要求
- Python 3.8+
- CUDA (推荐) 或 CPU
- 机器狗：云深处绝影 Lite3
- 感知主机：Jetson Nano/Orin/NX（或具有 RTSP 摄像头的主机）

### 安装依赖
```bash
pip install torch torchvision opencv-python numpy insightface mediapipe requests
```

### 运行
```bash
python main_controller1.py
```

### 键盘快捷键
| 按键 | 功能 |
|------|------|
| `m` | 手动切换模式（情绪↔手势） |
| `s` | 录入人脸 |
| `d` | 删除已录入人员 |
| `空格` | 紧急停止 |
| `q` | 退出系统 |

## 📁 项目结构
```
├── main_controller1.py           # 主程序入口
├── emotion_behavior_controller.py # 动作序列引擎
├── gesture_control.py             # 手势识别控制器
├── depth_guard_client.py          # 深度相机安全检测客户端
├── udp_client.py                  # UDP 通信底层封装
├── models/
│   ├── __init__.py
│   ├── vgg.py                     # VGG19 模型
│   ├── resnet.py                  # ResNet 模型
│   └── buffalo_m/                 # InsightFace 人脸模型
├── transforms/                    # 图像预处理
├── FER2013_VGG19/                 # VGG19 预训练权重
└── hand_landmarker.task           # MediaPipe 手势模型
```

### 手动下载大模型文件

以下 2 个模型文件超过 GitHub 100MB 限制，需要手动下载并放入对应目录：

| 文件 | 大小 | 下载地址 | 放置路径 |
|------|------|---------|---------|
| `1k3d68.onnx` | 143MB | [InsightFace GitHub](https://github.com/deepinsight/insightface/releases) | `models/buffalo_m/1k3d68.onnx` |
| `w600k_r50.onnx` | 174MB | [InsightFace GitHub](https://github.com/deepinsight/insightface/releases) | `models/buffalo_m/w600k_r50.onnx` |

或者运行以下命令自动下载：
```bash
# 安装 insightface 后会提示下载 buffalo_m 模型
pip install insightface
python -c "import insightface; insightface.model_zoo.get_model('buffalo_m')"
```

## 🔧 配置参数
关键配置在 `main_controller1.py` 顶部：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ROBOT_IP` | `192.168.2.1` | 机器狗控制 IP |
| `ROBOT_PORT` | `43893` | UDP 控制端口 |
| `CAMERA_SOURCE` | `rtsp://192.168.1.120:8554/test` | RTSP 视频流地址 |
| `EMOTION_COLLECT_SECONDS` | `5` | 情绪采样窗口(秒) |
| `FRAME_SKIP` | `3` | 帧跳过数 |

## 🔗 网络拓扑
```
Jetson (感知主机)
├── 网口 ─── 机器狗本体 (192.168.1.x)   →  UDP 控制
└── WiFi ─── 路由器/手机热点             →  可选外网访问
```

## 🙏 致谢
- [云深处科技](https://www.deeprobotics.cn/) — 绝影 Lite3 机器狗
- InsightFace — 人脸检测框架
- MediaPipe — 手势识别框架
