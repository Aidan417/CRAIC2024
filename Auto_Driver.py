import hashlib
import json
import os
from ctypes import *
import numpy as np
import cv2

import paddle.fluid as fluid
from PIL import Image
import paddle

paddle.enable_static()
import sys, os
import torch
import serial
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import (non_max_suppression)
from utils.torch_utils import select_device
from multiprocessing import Process, Queue
import time
from hex_change import car_drive

# 速度（初始速度）
vel = 1558
# 转向角（初始转向角）
angle = 1500
t = 0
num = 0
q = Queue()
ser = None


# 计算MD5值的函数
def calculate_md5(data):
    """
    计算输入数据的MD5哈希值
    :param data: 输入数据（字符串）
    :return: MD5哈希值（十六进制字符串）
    """
    md5 = hashlib.md5()
    md5.update(data.encode('utf-8'))
    return md5.hexdigest()


# 处理车道识别的函数
def lane():
    global vel
    global angle
    global t
    global t_end
    global num
    global ser

    def dataset(frame):
        """
        处理图像数据，提取车道信息
        :param frame: 输入图像
        :return: 处理后的图像数据
        """
        lower_hsv = np.array([0, 43, 46])
        upper_hsv = np.array([34, 255, 255])
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # 转换到HSV色彩空间
        mask = cv2.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)
        img = Image.fromarray(mask)
        img = img.resize((120, 120), Image.ANTIALIAS)
        img = np.array(img).astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = img.transpose((2, 0, 1))
        img = img[(2, 1, 0), :, :] / 255.0  # 归一化处理
        img = np.expand_dims(img, axis=0)
        return img

    # 加载模型
    save_path = "../model/model_infer/"
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    [infer_program, feeded_var_names, target_var] = fluid.io.load_inference_model(dirname=save_path, executor=exe)
    ser = serial.Serial('/dev/ttyACM0', 38400)  # 串口通信初始化
    time.sleep(1)
    cap = cv2.VideoCapture('/dev/cam_lane')  # 车道摄像头初始化
    while True:
        ret, frame = cap.read()
        if ret:
            img = dataset(frame)
            result = exe.run(program=infer_program, feed={feeded_var_names[0]: img}, fetch_list=target_var)
            angle = result[0][0][0]
            angle = int(angle + 0.5)

            # 限制转向角范围
            if angle < 1100:
                angle = 1100
            if angle > 1900:
                angle = 1900

            # 如果有速度信息且速度为1505，则初始化控制
            if not q.empty() and num == 0:
                vel = q.get()
                if vel == 1505:
                    num = 1
                ser.write(car_drive(vel, 1500))

            elif not q.empty():
                a = q.get()
                pass

            # 速度为1505且状态为1时，延迟切换到1552
            if vel == 1505 and num == 1:
                time.sleep(1.2)
                vel = 1552
                num = 2
                ser.write(car_drive(vel, angle))

            # 当状态为2时，执行控制
            if num == 2:
                t += 1
                if int(t) >= 70:
                    num = 0
                    t = 0
                    t_end = 0
                    vel = 1552

            # 写入控制指令
            ser.write(car_drive(vel, angle))
            cv2.imshow('lane', frame)
            if cv2.waitKey(1) == 27:
                ser.write(car_drive(1500, 1500))
                cv2.destroyAllWindows()
                cap.release()
                sys.exit(0)
                break
        else:
            print('lane相机打不开')


# 处理交通标志识别的函数
def sign():
    global vel
    global angle
    device = select_device('cpu')
    half = device.type != 'cpu'  # 仅CUDA支持半精度
    weights = '../model/yolov5_model/best.pt'

    # 加载模型
    model = attempt_load(weights, map_location=device)  # 加载FP32模型
    names = model.module.names if hasattr(model, 'module') else model.names
    cap = cv2.VideoCapture('/dev/cam_sign')  # 交通标志摄像头初始化
    print('打开相机')

    # 日志文件路径（可以根据需要更改路径）
    log_file_path = 'event_log.json'

    # 如果日志文件不存在，则创建它
    if not os.path.exists(log_file_path):
        with open(log_file_path, 'w') as f:
            f.write("[]")  # 初始化为空的JSON数组

    while True:
        ret, image = cap.read()
        if ret:
            # 预处理图像
            with torch.no_grad():
                img = letterbox(image, new_shape=320)[0]
                img = img[:, :, ::-1].transpose(2, 0, 1)  # 将BGR转换为RGB，并调整为3x416x416
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()
                img /= 255.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                pred = model(img, augment=False)[0]
                pred = non_max_suppression(pred, 0.4, 0.5, classes=False, agnostic=False)

                for i, det in enumerate(pred):
                    for *xyxy, conf, cls in reversed(det):
                        coor = []
                        label = names[int(cls)]
                        conf = round(float(conf), 2)
                        if conf >= 0.80:
                            for i in xyxy:
                                i = i.tolist()
                                i = int(i)
                                coor.append(i)

                            # 生成MD5哈希值
                            event_data = {
                                "label": label,
                                "confidence": conf,
                                "area": (coor[2] - coor[0]) * (coor[3] - coor[1])
                            }
                            event_json = json.dumps(event_data)
                            event_md5 = calculate_md5(event_json)

                            # 将事件详情和MD5哈希值输出到终端
                            print(f"事件: {event_json}, MD5: {event_md5}")

                            # 将事件数据写入日志文件
                            with open(log_file_path, 'r+') as f:
                                # 读取现有的数据
                                logs = json.load(f)
                                # 将新事件添加到日志列表
                                logs.append({"event": event_data, "md5": event_md5})
                                # 将更新后的日志数据写回文件
                                f.seek(0)  # 移动到文件开头
                                json.dump(logs, f, indent=4)

                            # 根据标签执行相应逻辑
                            area = event_data["area"]
                            if label == 'limit_10' and area >= 1350:
                                print('限速')
                                vel = 1545
                                q.put(vel)
                            elif label == 'crossing' and area >= 1200:
                                print('人行道')
                                vel = 1505
                                q.put(vel)
                            elif label == 'cancel_10' and area >= 1350:
                                print('限速解除')
                                vel = 1555
                                q.put(vel)
                            elif label == 'change_lanes':
                                print('变道')
                                pass
                            elif label == 'turn_left':
                                print('左转')
                                pass
                            elif label == 'turn_right':
                                print('右转')
                                pass
                            elif label == 'paper_red' and 5200 >= area >= 1200:
                                print('红灯')
                                vel = 1495
                                q.put(vel)
                            elif label == 'paper_greend' and area >= 250:
                                print('绿灯')
                                vel = 1558
                                q.put(vel)
                            elif label == 'warning' and area >= 300:
                                print('警告')
                                vel = 1555
                                q.put(vel)
                        del coor

            cv2.imshow('sign', image)
            k = cv2.waitKey(1)
            if k == 27:
                cv2.destroyAllWindows()
                cap.release()
                sys.exit(0)
                break
        else:
            print('sign相机打不开')


if __name__ == '__main__':
    # 启动车道和交通标志识别进程
    lane_run = Process(target=lane)
    sign_run = Process(target=sign)

    lane_run.start()
    sign_run.start()
