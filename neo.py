import time
import board
import atexit
import neopixel

# NeoPixels must be connected to D10, D12, D18 or D21 to work.
pixel_pin = board.D10
# The number of NeoPixels
num_pixels = 12
# The order of the pixel colors - RGB or GRB. Some NeoPixels have red and green reversed!
ORDER = neopixel.GRB

pixels = neopixel.NeoPixel(pixel_pin, num_pixels, brightness=1, auto_write=False, pixel_order=ORDER)

# 종료 시 리소스 해제
def handle_exit():
    neoOn("off")
    # allOff()
    print("[handle_exit] exit.")

def neoOn(color):
    pixels.fill(color)
    pixels.show()
    print(f"[neo] on rgb {color}")
    # if color == "red":
    #     pixels.fill((255, 0, 0))
    #     print("[neo] redOn")
    # elif color == "green":
    #     pixels.fill((0, 255, 0))
    #     print("[neo] greenOn")
    # elif color == "blue":
    #     pixels.fill((0, 0, 255))
    #     print("[neo] blueOn")
    # elif color == "lightblue":
    #     pixels.fill((126, 132, 247))
    #     print("[neo] lightblueOn")
    # elif color == "white":
    #     pixels.fill((255, 255, 255))
    #     print("[neo] whiteOn")
    # # elif color == "off":
    # elif color == "test":
    #   test = (255,255,255)
    #   pixels.fille(test)
    #   print("[neo] test")
    # else:
    #     pixels.fill((0, 0, 0))
    #     print("[neo] not Found Color")


def neoOff():
  pixels.fill((0, 0, 0))
  print("[neo] alloff")
  pixels.show()
#종료 시 실행되는 함수를 호출해둠

if __name__ == '__main__':
    atexit.register(handle_exit)
    print("[neo] Start.")
    neoOn("off")
    # allOff()
    time.sleep(1)
    while True:
        neoOn("red")
        neoOn("green")
        neoOn("blue")
        neoOn("off")
        '''
		redOn()
		time.sleep(1)
		allOff()
		time.sleep(1)
		greenOn()
		time.sleep(1)
		allOff()
		time.sleep(1)
		blueOn()
		time.sleep(1)
		allOff()
		time.sleep(1)
		'''
'''
def redOn():
	pixels.fill((255, 0, 0))
	pixels.show()
	print("[neo] redOn")
	# time.sleep(1)

def greenOn():
	pixels.fill((0, 255, 0))
	pixels.show()
	print("[neo] greenOn")
	# time.sleep(1)

def blueOn():
	pixels.fill((0, 0, 255))
	pixels.show()
	print("[neo] blueOn")
	# time.sleep(1)

def allOff():
	pixels.fill((0, 0, 0))
	pixels.show()
	print("[neo] allOff")
	# time.sleep(1)
'''