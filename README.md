# YOLO 目标跟随系统

摄像头实时检测 → 鼠标点选目标 → 自动跟踪 → 输出跟随指令。

input: 鼠标点击事件
output: 距离指令，方向指令

## 功能特性
- 实时目标检测（YOLOv8）
- 鼠标点击选中目标
- IoU 帧间跟踪 + 丢失处理
- 输出方位/距离指令（Left/Right/Forward/Backward/Stop）

## 环境依赖
- Python 3.x
- 依赖库见 requirements.txt
- yolov8n.pt 模型（需手动下载，下载地址：https://...）

## 安装
1. 建虚拟环境
2. pip install -r requirements.txt
3. 下载 yolov8n.pt 放到项目目录

## 使用
python test01.py
- 鼠标点击画面中的物体 = 选中它
- 按 q = 退出

## 工作原理
detect → update_tracking → update_metrics → make_command → draw_*

## 备注
在deepseekv4flash + Claude Code协助下完成
