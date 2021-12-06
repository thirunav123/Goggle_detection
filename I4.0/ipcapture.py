import cv2

#print("Before URL")
cap = cv2.VideoCapture('rtsp://admin:Rane@123@192.168.0.109/')
# cap = cv2.VideoCapture(0)

#print("After URL")

while True:

    #print('About to start the Read command')
    ret, frame = cap.read()
    #print('About to show frame of Video.')
    cv2.imshow("Capturing",frame)
    #print('Running..')

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()