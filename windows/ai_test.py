##########################################################################
# AI
##########################################################################
import time
import cv2
from ultralytics import YOLO

find_plate_model = YOLO("E:/1.Project/1.Thub/3.AISolution/1.scr/plate_detection/model/find_plate_240123.pt")
plate_char_model = YOLO("E:/1.Project/1.Thub/3.AISolution/1.scr/plate_detection/model/plate_char_240118.pt")

classes_type = ['plate1', 'plate2']
classes = ['1','2','3','4','5','6','7','8','9','0',
           '가', '나', '다', '라', '마',
           '거', '너', '더', '러', '머', '버', '서', '어', '저',
           '고', '노', '도', '로', '모', '보', '소', '오', '조',
           '구', '누', '두', '루', '무', '부', '수', '우', '주',
           '아', '바', '사', '자', '허', '하', '호', '배', 
           '서울', '서울', '부산', '부산', '대구', '대구', '인천', '인천', 
           '광주', '광주', '대전', '대전', '울산', '울산', '세종', '세종', 
           '경기', '경기', '강원', '강원', '충북', '충북', '충남', '충남', '전북', '전북', 
           '전남', '전남', '경북', '경북', '경남','경남', '제주', '제주', 
           '육', '해', '공', '국', '합', '초', '퍼']
           

def detect_plate_v8(img_path) :
    now_AI = time
    time_AI = now_AI.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[detect_plate_v8] Start. {time_AI}")
    frame = cv2.imread(img_path, cv2.IMREAD_COLOR)
    #현재(ver.01) 카메라 180도 회전되어 달려있음 -> 프로그램으로 회전시켜줌
    #frame = cv2.rotate(frame1, cv2.ROTATE_180)
    #image 폴더에 raw.jpg로 저장
    cv2.imwrite('E:/1.Project/1.Thub/3.AISolution/1.scr/LPR/parking_v1-main/parking_v1-main/temp/raw.jpg',frame)
    
    results = find_plate_model(frame)  # inference

    obj_list = []

    for result in results:
        for box in result.boxes.data :
            x1, y1, x2, y2, conf, cls = int(box[0]), int(box[1]), int(box[2]), int(box[3]), float(box[4]),  int(box[5])
            if conf >= 0.8: # 0.5 이상으로 확신하는 경우만 바운딩 박스를 그림
                label = classes_type[cls]
                obj_list.append([conf, x1, y1, x2, y2])
                

    if obj_list :
        obj_list.sort(key= lambda x:x[0])
        x = obj_list[-1][1]
        y = obj_list[-1][2]
        h = obj_list[-1][4] - obj_list[-1][2]
        w = obj_list[-1][3] - obj_list[-1][1]
        crop_img = frame[y : y + h, x : x + w]
        
        now_AI = time
        time_AI = now_AI.strftime('%Y-%m-%d %H:%M:%S')
        print(f"[detect_plate_v8] End. {time_AI}")
        return crop_img, x1, y1, x2, y2, cls
        
    else :
        print("[detect_plate_v8] 번호판이 감지되지 않았습니다.")

def detect_char_v8(frame, plate_type):
    obj_list = []

    results = plate_char_model(frame)  # inference

    for result in results:
        for box in result.boxes.data :
            x1, y1, x2, y2, conf, cls = int(box[0]), int(box[1]), int(box[2]), int(box[3]), float(box[4]),  int(box[5])
            if conf >= 0.7: # 0.9 이상으로 확신하는 경우만 바운딩 박스를 그림
                label = classes[cls]
                obj_list.append([label, x1, y1, x2, y2])


    if obj_list :        
        
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