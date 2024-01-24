import subprocess

def capture() :
    print("[capture] Start")
    subprocess.call(['sh', '/home/pi/workspace/parking_v1/script/capture.sh'])
    '''
    capture.sh 내용 :
    cd /home/pi/workspace/parking_v1
    libcamera-jpeg -o image/raw.jpg --rotation 180 --width 1920 --height 1080 -n -t 100
    '''
    print("[capture] Image Capture finish")