#main.py
#to test pins on Rasperry Pi Pico
print("Starting.")

from machine import Pin
import time

#test LEDs
ledRed = Pin(16, Pin.OUT)
ledGreen = Pin(18, Pin.OUT)
ledBlue = Pin(26, Pin.OUT)
ledYellow = Pin(28, Pin.OUT)


ledRed.toggle()
ledGreen.toggle()
ledBlue.toggle()
ledYellow.toggle()


time.sleep(1)

ledRed.toggle()
ledGreen.toggle()
ledBlue.toggle()
ledYellow.toggle()

ledRed.on()


#test switches
SW1 = Pin(14, Pin.IN, Pin.PULL_UP)
SW2 = Pin(15, Pin.IN, Pin.PULL_UP)

while True:
    if SW1.value() == 1:
        ledBlue.toggle()
        
    if SW2.value() == 0:
        ledYellow.toggle()
               

#main loop==================================================================
#while True:
    
    #read state of slider switches
#    nValueInput0 = slider1.value()
#   nValueInput1 = slider2.value()

#    print("infereing: {0:.0d}, {1:.0d}".format(nValueInput0, nValueInput1))
    

