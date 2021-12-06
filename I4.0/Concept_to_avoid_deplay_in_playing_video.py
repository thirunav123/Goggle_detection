import cv2
import numpy as np
import time
import threading

cap = cv2.VideoCapture(r'D:\videos\5.mp4')

x=50
y=50
w=640
h=640
def thread_task(a):
    global x,y
    for i in range(3):
        x += (i*50)
        y += (i*50)
        time.sleep(2)
t1 = threading.Thread(target=thread_task, args=(1,))
t1.start()

if (cap.isOpened()== False):
    print("Error opening video file")

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        print(t1.is_alive())
        if not t1.is_alive():
            t1 = threading.Thread(target=thread_task, args=(1,))
            print('Not alive, started')
            t1.start()
        else:
            print("Alive{0},{0}".format(x,y))
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
        cv2.imshow('Frame', cv2.resize(frame,(800,600)))
        # Press Q on keyboard to exit
        #time.sleep(.09)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    
    #time.sleep(1)

    # Break the loop
    else:
	    break

# When everything done, release
# the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
