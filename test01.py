#加载模型
import cv2
from ultralytics import YOLO
import time

#常量
ALPHA = 0.95 #指数滤波
IOU_THRESHOLD = 0.3   # IOU匹配阈值
DEAD_ZONE_RATIO = 0.02 #6 倍 ratio 噪声
TREND_THRESHOLD = 0.02 #4 倍 trend 静止抖动
LOST_FRAMES = 30

#定义变量
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0) #0表示默认摄像头
prev_time = time.time()
frame_size = ()
curr_detec = []
selected = {
    "bbox" : None,
    "name" : "",
    "active" : False,
    "setpoint" : None,
    "smooth" : None, #平滑后的ratio
    "lost_count" : 0
}

#函数声明
def compute_iou(boxA, boxB):
    #重叠区域的两个点
    inter_x1 = max(boxA[0],boxB[0])
    inter_y1 = max(boxA[1],boxB[1])
    inter_x2 = min(boxA[2],boxB[2])
    inter_y2 = min(boxA[3],boxB[3])
    #重叠面积计算
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_aera = inter_w * inter_h
    #各自面积
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    #并集面积
    union_aera = areaA + areaB - inter_aera
    if union_aera == 0:
        return 0.0
    return inter_aera / union_aera


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        #解决物体重叠的情况
        best = None
        best_area = float("inf")

        for det in curr_detec:
            x1, y1, x2, y2 = det["bbox"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                area = (x2 - x1) * (y2 - y1)
                if area < best_area:
                    best_area = area
                    best = det
        if best is not None:
            selected["bbox"] = list(best["bbox"])
            selected["name"] = best["name"]
            selected["active"] = True
            #计算并保存设定点的ratio
            bx1, by1, bx2, by2 = best["bbox"]
            box_area = (bx2 - bx1) * (by2 - by1)
            fw, fh = frame_size
            selected["setpoint"] = box_area / (fw * fh)
            selected["smooth"] = selected["setpoint"] #清除之前的污染
            selected["lost_count"] = 0 #选中成功后清零

            print(f"选中了： {best['name']}, 设定点ratio = {selected['setpoint']:.4f}")
        else:
            selected["active"] = False
            
#1 检测层
def detect(frame):
    #在此做检测
    objects = []
    results = model(frame, stream = True)
    for r in results:
        boxes = r.boxes
        if boxes is not None:
        #第一个循环：收集信息
            for box in boxes:
                x1,y1,x2,y2 = map(int,box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                name = r.names[cls_id]
                #存入本帧列表
                objects.append({"bbox" : [x1, y1, x2, y2], "name" : name,"conf" : conf})
    return objects

#2 追踪层
def update_tracking(objects, selected):

    if not selected["active"]:
        return -1

    best_idx = -1   
    best_iou = 0.0
    #寻找最佳iou
    for i, det in enumerate(objects):
        iou = compute_iou(selected["bbox"], det["bbox"])
        #类别一致性
        if iou > best_iou and det["name"] == selected["name"]:
            best_iou = iou
            best_idx = i
    if best_iou > IOU_THRESHOLD:
        selected["bbox"] = objects[best_idx]["bbox"]#更新为当前帧的新位置
        selected["lost_count"] = 0
        return best_idx
    else:
        #加入丢帧处理
        selected["lost_count"] += 1
        if selected["lost_count"] >= LOST_FRAMES:
            selected["active"] = False
            selected["lost_count"] = 0
        return -1

#3 测量层
def update_metrics(selected, frame_w, frame_h, frame_cx, frame_cy):
    
    x1, y1, x2, y2 = selected["bbox"]
    #方位
    target_cx = (x1 + x2) // 2  
    target_cy = (y1 + y2) // 2
    dx = target_cx - frame_cx
    dy = target_cy - frame_cy
    #距离
    cur_ratio = ((x2 - x1) * (y2 - y1)) / (frame_w * frame_h)   # 0.0 ~ 1.0，越大越近
    #对ratio滤波
    selected["smooth"] = ALPHA * selected["smooth"] + (1 - ALPHA) * cur_ratio
    trend = cur_ratio - selected["smooth"]
    #距离误差：正->太远 负->太近
    e = selected["setpoint"] - cur_ratio 
    return {"dx":dx, "dy":dy, "target_cx":target_cx, "target_cy":target_cy, "cur_ratio":cur_ratio, "trend":trend, "e":e}

#4 指令层
def make_command(metrics, frame_w):
    if not metrics:
        return "Stop"
    
    if abs(metrics["e"]) < DEAD_ZONE_RATIO:
        distance_cmd = "Stop"
    elif metrics["e"] > DEAD_ZONE_RATIO:
        distance_cmd = "Forward"
    else:
        distance_cmd = "Backward"
    #具有更高的优先级
    DEAD_ZONE_X = frame_w // 8
    #距离位置计算完后统一输出
    if abs(metrics["dx"]) > DEAD_ZONE_X:
        command = "Left" if metrics["dx"] < 0 else "Right"
    else:
        command = distance_cmd
    return  command

#5 显示层
def draw_boxes(frame, objects, best_idx):
    for i, det in enumerate(objects):
        x1, y1, x2, y2 = det["bbox"]
        #判断是否是选中的目标
        if i == best_idx:
            color = (0, 0, 255) #红色
        else:
            color = (0,255, 0) #绿色
        #在此显示和输出
        #画矩形框
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        #写标签文字
        label = f"{det['name']} {det['conf']:.2f}"
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (255,0,0),1)

def draw_hud(frame, fps, metrics, command, selected):
    #将检测后的一帧显示
    #写信息
    info_lines = [f"FPS: {fps:.1f}"]
    if metrics:
        info_lines.append(f"Target: {selected['name']}")
        info_lines.append(f"Deviation: dx={metrics['dx']:+d}, dy={metrics['dy']:+d}") 
        info_lines.append(f"Command: {command}")
        #关于ratio的显示
        info_lines.append(f"Ratio: {metrics['cur_ratio']:.4f}")
        info_lines.append(f"Smooth: {selected['smooth']:.4f}")
        info_lines.append(f"Trend: {metrics['trend']:+.4f}")        # ← 带正负号，看方向
        info_lines.append(f"Setpoint: {selected['setpoint']:.4f}")
    else:
        info_lines.append("Target: None")
            
    for i, line in enumerate(info_lines):
        cv2.putText(frame, line, (10, 30 + i * 25),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def draw_guides(frame, metrics, frame_cx, frame_cy, frame_w, frame_h):
    if not metrics:
        return
    # 目标中心点（红点）
    cv2.circle(frame, (metrics["target_cx"], metrics["target_cy"]), 6, (0, 0, 255), -1)
    # 画面中心十字线（白色）
    cv2.line(frame, (frame_cx, 0), (frame_cx, frame_h), (255, 255, 255), 1)
    cv2.line(frame, (0, frame_cy), (frame_w, frame_cy), (255, 255, 255), 1)
    # 目标中心 → 画面中心的连线（青色），直观显示偏差方向
    cv2.line(frame, (metrics["target_cx"], metrics["target_cy"]), (frame_cx, frame_cy), (255, 255, 0), 2)
    


cv2.namedWindow("Camera")
cv2.setMouseCallback("Camera", on_mouse)
    
def main():
    global prev_time, frame_size, curr_detec
    while True:
        #在此记录时间
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        #在此读取摄像头
        ret, frame  = cap.read()
        if not ret:
            break
        #确定位置信息
        frame_h, frame_w = frame.shape[:2]
        frame_cx = frame_w // 2
        frame_cy = frame_h // 2
        frame_size = (frame_w, frame_h)
        #1 检测
        objects = detect(frame)
        #鼠标回调函数需要
        curr_detec = objects
        #2 跟踪
        best_idx = update_tracking(objects, selected)
        #3 计算
        metrics = update_metrics(selected, frame_w, frame_h, frame_cx, frame_cy) if best_idx >= 0 else None
        #4 指令
        command = make_command(metrics, frame_w)
        #5 画框
        draw_boxes(frame, objects, best_idx)
        draw_hud(frame, fps, metrics, command, selected)
        draw_guides(frame, metrics, frame_cx, frame_cy, frame_w, frame_h)
            
        cv2.imshow("Camera",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

