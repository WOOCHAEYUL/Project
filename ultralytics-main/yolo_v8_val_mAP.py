import numpy as np
import os
import cv2
import time
from ultralytics import YOLO

from matplotlib import font_manager, rc # 폰트 세팅을 위한 모듈 추가
font_path = "C:/Windows/Fonts/malgun.ttf" # 사용할 폰트명 경로 삽입
font = font_manager.FontProperties(fname = font_path).get_name()
rc('font', family = font)


#model = YOLO("./runs/detect/train/weights/last.pt")
model = YOLO('E:/1.Project/1.Thub/3.AISolution/1.scr/ultralytics-main/ultralytics-main/runs/detect/train/weights/best.pt') #YOLO("E:/1.Project/1.Thub/3.AISolution/1.scr/ultralytics-main/ultralytics-main/runs/detect/train22/weights/best.pt")


if __name__ == '__main__':
    # Evaluating the model on the test dataset
    #metrics = model.val(conf = 0.25, data='./ultralytics/cfg/datasets/defect.yaml')
    #metrices = model.val(conf=0.9, split='test')
    #metrics = model.val(conf = 0.5, data='E:/1.Project/1.Thub/3.AISolution/1.scr/ultralytics-main/ultralytics-main/dataset/disabled__person/dp.yaml')
    metrics = model.val(conf = 0.75, data='E:/1.Project/1.Thub/3.AISolution/1.scr/ultralytics-main/ultralytics-main/dataset/coco/coco.yaml')