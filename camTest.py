from picamera import PiCamera
from time import sleep

camera = PiCamera()

camera.start_preview()

with open("/home/pi/Desktop/dart_test_images/dart_test_images.txt", "w") as f:
    count = 0
    while True:
        pie = input('Pie (or b to stop):')
        mult = input('Multiplier (1,2,3):')
        
        if pie == 'b':
            break
        
        camera.capture('/home/pi/Desktop/dart_images/' + str(pie) + '_' + str(mult) + '_' + str(count) + '.jpg')
        f.write(str(pie) + '_' + str(mult) + '_' + str(count) + '\n')
        count += 1
camera.stop_preview()