import cv2
import os
import numpy as np

img = cv2.imread(r'D:\images - all high resolution\train\frame248.jpg')
x=200
y=1100
w=640
h=640
ip=r'D:\images - all high resolution\combined'
ep=r'D:\Cropped_images'
count=1
for i in range(5):
    try:
        import_path = os.path.join(ip,'frame{}.jpg'.format(str(i)))
        #print(import_path)
        img = cv2.imread(import_path)
        cropped_image = img[x:x+w, y:y+h]
        export_path = os.path.join(ep,'img{}.jpg'.format(str(count)))
        print(export_path)
        count=count+1
        #cv2.imwrite(export_path, cropped_image)
        cv2.imshow("img",cropped_image)
        cv2.waitKey(10)
    except:
        pass


# Display cropped image
#cv2.imshow("cropped", cropped_image)
#cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 3)
#cv2.imshow("original", img)

# Save the cropped image
#cv2.imwrite("Cropped Image.jpg", cropped_image)

#cv2.destroyAllWindows()