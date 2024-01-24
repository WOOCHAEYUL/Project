'''
# 대기 : waiting  : A0105
# 입차 : entrance : A0101 : 라이다 인식거리 <= 500cm & 번호판 검출
# 정차 : stop     : A0102 : 라이다 인식거리 <= 주차 거리 -> 5초간 거리 변화 없음
# 주차 : parking  : A0103 : 정차 시 촬영한 사진의 번호판 좌표 = 1분 뒤 촬영한 사진의 번호판 좌표
# 출차 : exit     :"A0104": 500cm <= 라이다 인식거리 & 번호판 검출 안됨
'''

#import neo
import json
import time
import atexit
import serial
import shutil
import filecmp
#import schedule
#import getmac
import requests
import ai_test
#import mqtt
import camera as cam
import neo_color as light
from threading import Thread
#from pydub import AudioSegment
#from pydub.playback import play

"""
##########################################################################
# 상수
##########################################################################
LIMIT_PARKING = 150
DISTANCE_ENTRANCE = 200    # 입차 범위(cm)
DISTANCE_STOP = 100    # 정차 범위(cm)
INTERVAL_STOP = 5       # 입차 -> 정차 라이더 비교 시간 간격, 2 = 2초
INTERVAL_PARKING = 10   # 정차 -> 주차 카메라 비교 시간 간격, 30 = 30초
INTERVAL_LIVE = 5   # live 신호 MQTT 송신 주기, 1 = 1분
VERSION_FW = "1.0"
##########################################################################
# 변수초기화
##########################################################################
# 내가 쓰는 변수
mode_state = "A0104" #대기
plate_number = ""
already_waiting = False
already_entrance = False
already_stop = False
already_parking = False
already_exit = False
time_before_stop = time.time()
flag_camera = True
source_path = "/home/pi/workspace/parking_v1/image/raw.jpg"
lidar_distance = 1000
violation = "N"
##########################################################################
# API 선언
##########################################################################
# 사용가능한 서버 리스트
serverUrl = {
    'staging': "http://devpsms.thubiot.com",
    'sta' :"192.168.0.17:14100",
    'prod': "http://psms.thubiot.com",
    'local': "localhost:14100",
    'ok_pc': "192.168.0.236:14100"
}
# 현재 프로그램에서 사용하는 서버
server = serverUrl['sta']
# API url
url_saveParkingOperation = "http://" + server + "/api/psms/saveParkingOperation"
url_updateFwVersion = "http://" + server + "/api/psms/updateFwVersion"
url_saveDeviceInterface = "http://" + server + "/api/psms/saveDeviceInterface"
url_getCmnCd = "http://"+server+"/api/psms/getCmnCd"

##########################################################################
# Serial 선언
##########################################################################
buad = 115200
port = '/dev/ttyUSB0'
ser = serial.Serial(port, buad, timeout=3, parity = serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE)
##########################################################################
# Audio
##########################################################################
audio = AudioSegment.from_file("/home/pi/workspace/parking_v1/media/warning.m4a", format="m4a")
audio_announcement = AudioSegment.from_file("/home/pi/workspace/parking_v1/media/announcement.m4a", format="m4a")

##########################################################################
# LiDar
##########################################################################
# 거리 값 측정
def read_LiDar(ser):
    global lidar_distance
    
    while True:
        data = ser.read(9)
        l_data = list(data)
        if l_data[0] == 0x59 & l_data[1] == 0x59:
            lidar_distance = l_data[3] * 255 + l_data[2]        

# 값을 바로 가져다 쓸 때 사용
def get_lidar_distance():
    global lidar_distance
    return lidar_distance
######################################################################################################
# 모드 선택
######################################################################################################
def init_wait():
  global already_entrance 
  global already_parking
  global already_stop
  global mode_state
  global violation

  mode_state = "A0104"
  already_entrance = False
  already_parking = False
  already_stop = False
  violation = "N"

def init_entrance():
  global already_entrance

  already_entrance = True
  # play(audio_announcement)
  API_transmit("saveParkingOperation")

def init_stop():
  global already_stop
  API_transmit("saveParkingOperation")
  time_before_stop = time.time()
  already_stop = True
  # neo.neoOn("green")

def init_parking():
  global already_parking
  API_transmit("saveParkingOperation")
  already_parking = True
"""
source_path = 'E:/1.Project/1.Thub/3.AISolution/1.scr/LPR/parking_v1-main/parking_v1-main/raw/raw.jpg'# "/home/pi/workspace/parking_v1/image/raw.jpg"
def state_machine():
  #global mode_state
  global plate_number
  global already_waiting
  global already_entrance
  global already_stop
  global already_parking
  global already_exit
  global time_before
  global time_before_stop
  global lidar_distance
  global flag_camera
  global data_before
  import cv2
  
  mode_state = "A0101"
  
  #cam.capture() # 사진을 촬영하여 image/raw.jpg 저장
  
  cam = cv2.imread(source_path)
  
  cv2.imshow('cam', cam)
  cv2.waitKey(0)
  #shutil.copy(source_path, "E:/1.Project/1.Thub/3.AISolution/1.scr/LPR/parking_v1-main/parking_v1-main/raw/" + mode_state)
  
  crop_result = ai_test.detect_plate_v8(source_path)  # 번호판의 유무를 저장된 사진으로 판단
  
    # 차량, 번호판 있음
  if crop_result is not None :
    crop, x1, y1, x2, y2, plate_type = crop_result
    print('plate_type',plate_type) 
    # 차량 번호를 인식함        
    plate_number = ai_test.detect_char_v8(crop, plate_type)
    print(f"[state_machine]-[waiting] 인식된 차량 번호 : {plate_number}, 좌표 : {x1}, {y1}, {x2}, {y2}")
    
    mode_state = "A0101" # 입차 모드로 전환
    # 파일 복사
    shutil.copy(source_path, "E:/1.Project/1.Thub/3.AISolution/1.scr/LPR/parking_v1-main/parking_v1-main/raw/" + mode_state)
    #play(audio_announcement)
    #already_exit = False
    #flag_camera = True
  # 차량이 아님
  else:
    #flag_camera = False
    #already_exit = False
    mode_state = "A0104" # 차량이 아니면 출차상태로
  return "리턴 : 입차시 차량 판별"

"""
  if get_lidar_distance() >= DISTANCE_ENTRANCE : 
    # 대기모드 = 주차가능
    # 라이다 거리가 입차기준보다 크다.
    # time.sleep(1)
    if mode_state != "A0104" :
      print("########## 출차상태 ###########")
      init_wait()
      cam.capture() # 사진을 촬영하여 image/raw.jpg 저장
      shutil.copy(source_path, "/home/pi/workspace/parking_v1/image/" + mode_state)
      API_transmit("saveParkingOperation")
      # flag_camera = True
      already_exit = False
    elif already_exit == False:
      print("############### 초기 상태")
      neo.neoOn(light.blue)
      API_transmit("saveParkingOperation") 
      flag_camera = True 
      already_exit = True
      
    return "리턴 : 출차상태"

  # else:
  if get_lidar_distance() < DISTANCE_ENTRANCE and mode_state == "A0104":
    # 뭔가 주차면 들어왔는데 대기모드다
    # 사진 촬영하고 차량번호판 인식 ai 동작, 
    print(f"@@@@@@@@@@@@ 물체 접근@@@@@@@@@@@ ::: mode_state =>  {mode_state}")
    
    if flag_camera == True:  # 카메라 촬영가능
      print("입차 사진 촬영")
      cam.capture() # 사진 촬영 및 image/raw.jpg 로 저장
      crop_result = ai.detect_plate_v8("image/raw.jpg")  # 번호판의 유무를 저장된 사진으로 판단

      # 차량, 번호판 있음
      if crop_result is not None :
        crop, x1, y1, x2, y2, plate_type = crop_result 
        # 차량 번호를 인식함        
        plate_number = ai.detect_char_v8(crop, plate_type)
        print(f"[state_machine]-[waiting] 인식된 차량 번호 : {plate_number}, 좌표 : {x1}, {y1}, {x2}, {y2}")
        
        mode_state = "A0101" # 입차 모드로 전환
        # 파일 복사
        shutil.copy(source_path, "/home/pi/workspace/parking_v1/image/" + mode_state)
        play(audio_announcement)
        already_exit = False
        flag_camera = True
      # 차량이 아님
      else:
        flag_camera = False
        already_exit = False
        mode_state = "A0104" # 차량이 아니면 출차상태로
    return "리턴 : 입차시 차량 판별"
  
  if mode_state == "A0101":
    # 입차모드 진입 시, 한번만 수행
    print(f"already_entrance :: {already_entrance}")
    if already_entrance == False:
      print("[state_machine] 입차 모드로 진입")
      init_entrance()
    else:
      # if get_lidar_distance() <= DISTANCE_STOP:
      print("입차모드에서 주차기준 안 ----")
      if flag_camera == True:
        print(" 사진 촬영")
        flag_camera = False
        cam.capture() # 사진을 촬영하여 image/raw.jpg 저장
        shutil.copy(source_path, "/home/pi/workspace/parking_v1/image/" + mode_state)
        time_before = time.time()
        data_before = get_lidar_distance()
        
      time_after = time.time()
      # INTERVAL_STOP동안 라이다 값이 변화가 없다면
      print(f"{time_before} :::: {time_after}")
      if time_after - time_before >= INTERVAL_STOP:
        print(f"입차 후 interval 정차 시간 동안 || {data_before} ::: {get_lidar_distance()}")

        if abs(get_lidar_distance()-data_before) < 3:
          print(f"정차시 라이다 거리 변화가 없다 :: 거리 차이 {abs(get_lidar_distance()-data_before)}")
          time_before = time_after
          already_entrance = False
          mode_state = "A0102"
          flag_camera = True
        else: # INTERVAL_STOP동안 라이다 값이 변했다면
          print("라이다 거리변화가 있다.")
          time_before = time_after
          data_before = get_lidar_distance()
    return "리턴 : 입차 차량판별 후 정차판별"
  #정차 모드 : stop : A0102
  if mode_state == "A0102":
    # 차가 출차기준까지 빠져버렸다면 출차모드
    if get_lidar_distance() >= DISTANCE_ENTRANCE:
      already_stop = False
      mode_state = "A0104"

    # 정차모드 진입 시, 한번만 수행
    if already_stop == False:
      print("[state_machine] 정차 모드로 진입")
      init_stop()
      time_before = time.time()
      data_before = get_lidar_distance()

    # 정차모드 루프
    else:
      print("[state_machine] 정차 모드 loop")
      if flag_camera == True:
        cam.capture() # 사진을 촬영하여 image/raw.jpg 저장
        shutil.copy(source_path, "/home/pi/workspace/parking_v1/image/" + mode_state)
        flag_camera = False
          
      # else:
      time_after = time.time()

      # INTERVAL_STOP동안 라이다 값이 변화가 없다면
      if time_after - time_before >= INTERVAL_PARKING:
        print(f"정차후 후 interval 주차 시간 동안 || {data_before} ::: {get_lidar_distance()}")
        if abs(get_lidar_distance()-data_before) <3:
          print(f"주차시 라이다 거리 변화가 없다 :: 거리 차이 {abs(get_lidar_distance()-data_before)}")
          time_before = time_after
          already_entrance = False
          mode_state = "A0103"
    
      # INTERVAL_STOP동안 라이다 값이 변했다면
        else:
          print("주차시 라이다 거리 변화가 있다")
          time_before = time_after
          data_before = get_lidar_distance()
          flag_camera = True
  # mode : parking  : A0103
  if mode_state == "A0103":
    if already_parking == False: # 주차모드 진입 시, 한번만 수행
      print("[mode_state] 주차 모드로 진입")
      init_parking()
    else:
      print("[mode_state] 주차 모드 loop")
      if get_lidar_distance() >= DISTANCE_ENTRANCE:
        if flag_camera == True:
          cam.capture() # 사진을 촬영하여 image/raw.jpg 저장
          shutil.copy(source_path, "/home/pi/workspace/parking_v1/image/" + mode_state)
          flag_camera = False
          mode_state = "A0104"
          already_parking = False
          
    return 0
  # mode : exit     :"A0104" 
  if mode_state == "A0104":
    if already_exit == False:
    # 출차모드 진입 시, 한번만 수행
      print("[state_machine] 출차 모드로 진입")
      already_exit = True
    return 0
"""
######################################################################################################
# API
######################################################################################################
"""def API_transmit(URL):  

  global violation

  print("[API_transmit] start")
  payload = {'deviceCd' : mqtt.mac}

  # way) 호출 정의
  if URL == "saveParkingOperation":
    print("[API_transmit]-[saveParkingOperation] Start")
    now = time
    time_API = now.strftime('%Y-%m-%d %H:%M:%S')
    # now = time.localtime()
    # strnow = time.strftime('%Y-%m-%d %H:%M:%S', now)

    headers = {'charset': 'utf8'}
    payload['parkingState'] = mode_state # 0, 1: 입차, 2: 정차, 3: 주차, 4:출차
    payload['regDt'] = time_API
    payload['carNumber'] = plate_number
    # files=[('file',('raw.jpg',open('/home/pi/workspace/parking_v1/image/'+ mode_state + '/raw.jpg','rb'),'image/jpg'))]
    files=[('file',('raw.jpg',open('/home/pi/workspace/parking_v1/image/raw.jpg','rb'),'image/jpg'))]

    # way) 정의를 토대로 호출
    try :
      print("[API] berfore API_transmit request")
      response = requests.request("POST", url_saveParkingOperation, headers=headers, data=payload, files=files)
      print("[API] after API_transmit request")

      response_data = response.json()
      print(f"[API] 호출 모드 : {mode_state}")

      print(f"response_data = {response_data}")
      #  response : {"success":true,"code":200,"msg":"성공하였습니다","data":{"carNumber":"123가4568","violation":"N","disabled":"Y"}}
      if mode_state != "A0104":
        if response_data['success']:
          violation = response_data['data']['violation']
          if response_data['data']['disabled'] == 'N':
            print("위반했다!!")
            # neo.neoOn("red")
            play(audio)
          elif response_data['data']['disabled'] == 'Y':
            print("장애인차량이다!!")
            neo.neoOn(light.green)

    except Exception as e:
      print("\n")
      print('↓↓↓Exception'*5)
      print(f"[API_transmit]-[saveParkingOperation] Exception : {e}")
      print('↑↑↑Exception'*5)
      print("\n")

    print(f"[API_transmit]-[saveParkingOperation] response : {response.text}")

  elif URL == "updateFwVersion":
    print("[API_transmit]-[saveParkingOperation] Start")

  # purpose) 장비등록 API URL     장비 등록 => 이 장비 본인이 알고있는 것, 처음 등록할 때 알 수 있는것
  elif URL == "saveDeviceInterface":
    # 여기 해당하는 API는 문자열로 변환해서 API 호출해야함
    print("[API_transmit]-[saveDeviceInterface] Start")

    payload['fwVersion'] = VERSION_FW
    print(f"[API_transmit]-[saveDeviceInterface] payload : {payload}")
    headers = {
      'charset': 'utf8',
      'Content-Type': 'application/json'
    }
    payload = json.dumps(payload)    
    try:    
      response = requests.request("POST", url_saveDeviceInterface, headers=headers, data=payload)
    except:
      print("[API_transmit]-[saveDeviceInterface] 정상 - 서버에 같은 값이 이미 있습니다.")
    print(f"[API_transmit]-[saveDeviceInterface] response : {response.text}")

######################################################################################################
# live신호 송신 및 스레드 타겟 함수                                                                                #
######################################################################################################
def schedule_live():
    try:
        mqtt.MQTT_transmit()
    except:
        pass

def scheduled_task():
    while True:
        schedule.run_pending()
        time.sleep(1)

def led_blink(color, interval):
  global violation

  while True:
    # print(f"led_blink violation value :: {violation}")
    time.sleep(0.2)
    if violation == "Y":
      neo.neoOn(color)
      time.sleep(interval)
      neo.neoOff()
      time.sleep(interval-0.2)
######################################################################################################
# 종료 시 리소스 해제                                                                                  #
######################################################################################################
def handle_exit():
    neo.neoOff()
    # allOff()
    print("[parking_main]-[handle_exit] exit.")
"""
######################################################################################################
# main                                                                                               #
######################################################################################################
if __name__ == '__main__' :
    print("[main] Start")
    """neo.neoOn(light.blue)
    atexit.register(handle_exit)

    API_transmit("saveDeviceInterface")
    mqtt.MQTT_transmit()

    schedule.every(INTERVAL_LIVE).minutes.do(schedule_live)
    
    scheuled_thread = Thread(target=scheduled_task, daemon=True)
    scheuled_thread.start()

    th_read_LiDAR = Thread(target=read_LiDar, args=(ser,), daemon=True)
    th_read_LiDAR.start()

    th_led_blink = Thread(target=led_blink, args=(light.red,0.6), daemon=True)
    th_led_blink.start()"""

    while True:
      state_result = state_machine()
      print(f"state result =======> {state_result}")
      time.sleep(1)