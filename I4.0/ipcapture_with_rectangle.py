import cv2

#print("Before URL")
cap = cv2.VideoCapture('rtsp://admin:Rane@123@192.168.0.109/')
# cap = cv2.VideoCapture(0)
x=50
y=50
w=640
h=640
#print("After URL")

while True:

    #print('About to start the Read command')
    ret, frame = cap.read()
    #print('About to show frame of Video.')
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)

    cv2.imshow("Capturing",frame)
    #print('Running..')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()