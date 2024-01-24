from threading import Thread
import cv2
from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import datetime
import atexit
from multiprocessing import Process
import subprocess
from sklearn.cluster import KMeans
from operator import itemgetter


def myPutText(src, text, pos, font_size, font_color) :
    img_pil = Image.fromarray(src)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype('fonts/gulim.ttc', font_size)
    draw.text(pos, text, font=font, fill= font_color)
    return np.array(img_pil)


COLOR = (0,128,0) # 흰색 
FONT_SIZE = 30


find_plate_model = YOLO("E:/1.Project/1.Thub/3.AISolution/1.scr/ultralytics-main/ultralytics-main/runs/detect/train9/weights/best.pt")
plate_char_model = YOLO("E:/1.Project/1.Thub/3.AISolution/1.scr/ultralytics-main/ultralytics-main/runs/detect/train7/weights/best.pt")

classes_type = ['plate1', 'plate2']
classes = ['1','2','3','4','5','6','7','8','9','0',
           '가', '나', '다', '라', '마',
           '거', '너', '더', '러', '머', '버', '서', '어', '저',
           '고', '노', '도', '로', '모', '보', '소', '오', '조',
           '구', '누', '두', '루', '무', '부', '수', '우', '주',
           '아', '바', '사', '자', '허', '하', '호', '배', 
           '서울', '서울', '부산', '부산', '대구', '대구', '인천', '인천', '광주', '광주', '대전', '대전', '울산', '울산', '세종', '세종', '경기', '경기', '강원', '강원', '충북', '충북', '충남', '충남', '전북', '전북', '전남', '전남', '경북', '경북', '경남','경남', '제주', '제주', '육', '해', '공', '국', '합', '초', '퍼']


def detect_plate_v8(img_path) :   
    frame = cv2.imread(img_path, cv2.IMREAD_COLOR)
    results = find_plate_model(frame)  # inference

    obj_list = []

    for result in results:
        for box in result.boxes.data :
            x1, y1, x2, y2, conf, cls = int(box[0]), int(box[1]), int(box[2]), int(box[3]), float(box[4]),  int(box[5])
            if conf >= 0.8: # 0.5 이상으로 확신하는 경우만 바운딩 박스를 그림
                obj_list.append([conf, x1, y1, x2, y2])

    if obj_list :
        obj_list.sort(key= lambda x:x[0])
        x = obj_list[-1][1]
        y = obj_list[-1][2]
        h = obj_list[-1][4] - obj_list[-1][2]
        w = obj_list[-1][3] - obj_list[-1][1]
        crop_img = frame[y : y + h, x : x + w]
        
        print('##########point##########', x1, y1, x2, y2)     
              
        return crop_img, x1, y1, x2, y2, cls
        
        
    else :
        print("list empty!")
        return None, None, None, None, None, None


def detect_char_v8(frame, plate_type):
    print('plate_type', plate_type)
    results = plate_char_model(frame)  # inference

    obj_list = []

    for result in results:
        for box in result.boxes.data :
            x1, y1, x2, y2, conf, cls = int(box[0]), int(box[1]), int(box[2]), int(box[3]), float(box[4]),  int(box[5])
            if conf >= 0.7: # 0.9 이상으로 확신하는 경우만 바운딩 박스를 그림
                label = classes[cls]
                obj_list.append([label, x1, y1, x2, y2])
                
    if obj_list :
        print('obj_list',obj_list)        
        
        if plate_type == 0:
            # x 값을 기준으로 정렬
            sorted_by_x = sorted(obj_list, key=lambda x: x[1])  
            print("x 값 기준 정렬: ", sorted_by_x)
            
            plate_number = ''
            for i in sorted_by_x:
                plate_number += i[0]

            plate_number= plate_number.replace(" ", "")
            print(plate_number)           
            
        elif plate_type == 1:
        
            top_line = [item for item in obj_list if item[2] < (max(item[2] for item in obj_list) + min(item[2] for item in obj_list)) / 2]
            bottom_line = [item for item in obj_list if item[2] >= (max(item[2] for item in obj_list) + min(item[2] for item in obj_list)) / 2]
            
            print('top_line', top_line)
            print('bottom_line', bottom_line)
            
            
            # 각 라인에서 y 좌표를 기준으로 정렬
            top_line.sort(key=lambda x: x[2])
            bottom_line.sort(key=lambda x: x[2])

            # 각 라인에서 x 좌표를 기준으로 정렬
            top_line.sort(key=lambda x: x[1])
            bottom_line.sort(key=lambda x: x[1])

            # 결과 출력
            result = top_line + bottom_line
            print('result::::',result)
            
            plate_number = ''
            for i in result:
                plate_number += i[0]

            plate_number= plate_number.replace(" ", "")
            print("plate_number", plate_number)
            
        return plate_number      
                
    else :
        print("list empty!")
        return None
        
    
def state_machine():

    filename = "C:/Users/user/Desktop/rasberrypi_video_save/plate_video/15_23-47-39.avi"
    cap = cv2.VideoCapture(filename)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    now = datetime.datetime.now().strftime("%d_%H-%M-%S")
    video = cv2.VideoWriter("./video/" + str(now) + ".avi", fourcc, 20.0, frame_size)

    while True:
        ret, img = cap.read()

        if not ret:
            print("No more frames in the video.")
            break

        img_copy = img.copy()

        cv2.imwrite('./capture_img/captured.jpg', img)
        print("Image captured")           

        crop, x1, y1, x2, y2, plate_type = detect_plate_v8('./capture_img/captured.jpg')

        if crop is not None:               
            plate_number = detect_char_v8(crop, plate_type)
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 128, 0), 3)
            img_copy = myPutText(img_copy, plate_number, (x1, y1 - 30), FONT_SIZE, COLOR)

            cv2.imwrite('./capture_img/captured_{}.jpg'.format(plate_number), img_copy)

            print('Plate number:', plate_number)
            print('x1, y1, x2, y2:', x1, y1, x2, y2)
            
        else:
            print("Crop is None")
            plate_number = None            
            
        cv2.imshow('img_copy', img_copy)
        cv2.waitKey(1)
        
        video.write(img_copy)
        
        key = cv2.waitKey(33)
        if key == ord("q"):break       

    cap.release()  # Release the video capture object                

if __name__ == '__main__':
    print("Start LPR Program")

    plate_list = []

    f = open('plate_list.txt', 'rt', encoding='utf-8')
    for line in f:
        line = line.strip()
        plate_list.append(line)
    f.close()

    print(plate_list)

    while True:
        state_machine()