import numpy as np
import os
import cv2
import time
from ultralytics import YOLO



#model = YOLO("./runs/detect/train/weights/last.pt")
model = YOLO('E:/1.Project/1.Thub/3.AISolution/1.scr/ultralytics-main/ultralytics-main/runs/detect/train2/weights/best.pt') #YOLO("E:/1.Project/1.Thub/3.AISolution/1.scr/ultralytics-main/ultralytics-main/runs/detect/train22/weights/best.pt")


if __name__ == '__main__':
    # Evaluating the model on the test dataset
    #metrics = model.val(conf = 0.25, data='./ultralytics/cfg/datasets/defect.yaml')
    #metrices = model.val(conf=0.9, split='test')
    #metrics = model.val(conf = 0.5, data='E:/1.Project/1.Thub/3.AISolution/1.scr/ultralytics-main/ultralytics-main/dataset/disabled__person/dp.yaml')
    metrics = model.val(data='E:/1.Project/1.Thub/3.AISolution/1.scr/ultralytics-main/ultralytics-main/dataset/plate_char/plate_char.yaml')
    print('metrics.box.map50', metrics.box.map50)
    print('metrics.box.map75', metrics.box.map75)
    print('metrics.box.map', metrics.box.map)
    print('metrics.box.maps', metrics.box.maps)
    
    