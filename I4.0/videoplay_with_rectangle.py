import cv2

#print("Before URL")
cap = cv2.VideoCapture(r'D:\Videos_SPR4.0\2021-11-24\1.mp4')
# cap = cv2.VideoCapture(0)
x=250
y=50
w=880
h=640
#print("After URL") 
while True:

    #print('About to start the Read command')
    ret, frame = cap.read()
    #print('About to show frame of Video.')
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)

    cv2.imshow("Capturing",cv2.resize(frame,(1280,720)))
    #cv2.imwrite(r'D:\1.jpg', frame)
    #print('Running..')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()