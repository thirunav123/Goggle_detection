import cv2,os

#print("Before URL")
cap = cv2.VideoCapture(r'D:\Videos_SPR4.0\2021-11-24\2.mp4')
# cap = cv2.VideoCapture(0)
x=250
y=50
w=880
h=640
ep=r'D:\Cropped_images_SPR_new'
count=0
i=534
#print("After URL")

while True:

    #print('About to start the Read command')
    ret, frame = cap.read()
    cropped_image = frame[ y:y+h,x:x+w]
    count=count+1

    if count%25==0:
        i=i+1
        export_path = os.path.join(ep,'img{}.jpg'.format(str(i)))
        cv2.imwrite(export_path, frame)
        print(export_path)
        

    #cv2.imshow("img",cropped_image)
    #cv2.waitKey(10)
    #print('About to show frame of Video.')
    #cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)

    #cv2.imshow("Capturing",frame)
    #print('Running..')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




