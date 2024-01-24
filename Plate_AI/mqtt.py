import time
import paho.mqtt.client as mqtt
import json
import getmac

##########################################################################
# MQTT 선언
##########################################################################
mac = getmac.get_mac_address()
mac = mac.replace(":", "")
topic_transmit = "psms/" + mac + "/live"
# MQTT 브로커의 주소와 포트
broker_address = "192.168.0.17"
broker_port = 1883

# 클라이언트 ID 및 인증 정보 설정
client_id = "parking_{}".format(mac)
username = "THUB_API_SERVER"  # MQTT 브로커에 등록된 사용자 이름
password = "ThubAPIServer!@#$"  # 사용자 비밀번호

# MQTT 클라이언트 생성
client = mqtt.Client(client_id)

# 인증 정보 설정
client.username_pw_set(username, password)

# 연결 이벤트 핸들러
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("[MQTT]-[on_connect] 연결 성공")
    elif rc == 1:
        print("[MQTT]-[on_connect] 연결 거부 - 프로토콜 버전이 허용되지 않음")
    elif rc == 2:
        print("[MQTT]-[on_connect] 연결 거부 - 클라이언트 식별자가 잘못되었음")
    elif rc == 3:
        print("[MQTT]-[on_connect] 연결 거부 - 브로커에 연결할 수 없음")
    elif rc == 4:
        print("[MQTT]-[on_connect] 연결 거부 - 연결 거부 - 사용자 이름 또는 비밀번호가 잘못됨")
    elif rc == 5:
        print("[MQTT]-[on_connect] 연결 거부 - 연결 거부 - 권한이 없음")

    client.subscribe("psms/" + mac +"/control")  # 구독할 토픽 설정

# 메시지 수신 이벤트 핸들러
def on_message(client, userdata, msg):
    message_control = msg.payload.decode()
    print(f"[MQTT]-[on_message] topic : {msg.topic}, payload : {message_control}")

    # JSON 형식의 문자열 파싱
    try:
        message_data = json.loads(message_control)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")

    # "type" 키가 있는 경우에만 처리
    if "type" in message_data:
        control_type = message_data["type"]

    # 램프 컨트롤
    if control_type == "lamp":
        value = message_data.get("value", "")
        print(f"[MQTT]-[on_message] Lamp control: {value}")
        
        if value == "redOn":
            print("[MQTT]-[on_message] redOn")
            neo.neoOn("red")
        elif value == "greenOn":
            print("[MQTT]-[on_message] greenOn")
            neo.neoOn("green")
        elif value == "blueOn":
            print("[MQTT]-[on_message] blueOn")
            neo.neoOn("blue")
        elif value == "neoOff":
            print("[MQTT]-[on_message] neoOff")
            neo.neoOn("off")

    # 스피커 컨트롤
    elif control_type == "speaker": #경고방송
        value = message_data.get("value", "")
        print(f"[MQTT]-[on_message] Speaker control: {value}")
        
        if value == "announce": #안내방송
            print("[MQTT]-[on_message] Announce")
            play(audio_announcement)
        elif value == "warning":
            print("[MQTT]-[on_message] Warning")
            play(audio)

# 이벤트 핸들러 등록
client.on_connect = on_connect
client.on_message = on_message
# MQTT 브로커에 연결
client.connect(broker_address, broker_port, 60)
# 루프 시작
client.loop_start()

# MQTT 송신
def MQTT_transmit():
    print("[MQTT_transmit] Start")
    now = time
    time_transmit = now.strftime('%Y-%m-%d %H:%M:%S')

    dict_payload = { "state" : "live",
                    "send_dt" : time_transmit }
    payload_live = json.dumps(dict_payload)

    try:
      print(f"mqtt Test :: {topic_transmit} || {payload_live}")  
      client.publish(topic_transmit,payload_live)
    except Exception as e :
        print(f"[MQTT_transmit] except : {e}")

    print(f"[MQTT_transmit] {time_transmit}, Published message")