# Jetson YOLOv6 Demo

在 NVIDIA Jetson（Orin / JetPack 6 + CUDA 12.6）上用 **YOLOv6 + CUDA** 做实时目标检测，并通过内网网页以 **720p MJPEG** 流式查看。

## 访问地址

服务默认监听 `0.0.0.0:8080`。本机常见可达地址：

| 网卡 | 地址 |
|------|------|
| ZeroTier (`ztu7mcmex3`) | `http://10.47.235.49:8080/` |
| 有线 (`eno1`) | `http://192.168.137.15:8080/` |

Mac / 同网设备浏览器打开上述任一地址即可看实时识别画面。

## 快速启动

```bash
cd ~/Jetson_Yolo_Demo
./run.sh
```

常用环境变量：

```bash
PORT=8080 CAMERA=0 ./run.sh
```

## 首次环境准备

```bash
# 系统依赖（只需一次）
sudo apt-get install -y cudss-cuda-12

# Python / 模型 / YOLOv6 源码
bash scripts/setup_deps.sh
```

说明：

- CUDA 版 PyTorch 来自 `https://pypi.jetson-ai-lab.io/jp6/cu126`
- 默认权重：`weights/yolov6n.pt`（YOLOv6-N，COCO）
- 摄像头默认 `/dev/video0`（UGREEN USB 2K 的采集节点）

## 项目结构

```
app/
  detector.py       # YOLOv6 CUDA 单帧推理
  stream_server.py  # Flask MJPEG 实时网页
scripts/
  setup_deps.sh
weights/            # 放置 yolov6n.pt
third_party/YOLOv6/ # Meituan YOLOv6（本地 clone，不入库）
run.sh
```

## API

- `GET /` — 实时页面
- `GET /video_feed` — MJPEG 视频流
- `GET /api/stats` — FPS / device 状态 JSON
- `GET /health` — 健康检查
