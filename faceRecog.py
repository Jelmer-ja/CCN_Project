# import numpy as np
import cv2
#
#catface_cascade = cv2.CascadeClassifier('C:/Users/Laura/Documents/TEMP/opencv/build/etc/haarcascades/haarcascade_frontalcatface.xml')
catface_cascade = cv2.CascadeClassifier('C:/Users/Laura/Documents/TEMP/opencv/build/etc/lbpcascades/lbpcascade_frontalcatface.xml')
#
# # load cat image
image = cv2.imread('C:/Users/Laura/Documents/TEMP/cats/images/Persian_164.jpg')
#image = cv2.imread('C:/Users/Laura/Documents/TEMP/cats/images/Ragdoll_165.jpg')
# #img = cv2.imread('C:/Users/Laura/Pictures/Gala Histos 2015/11210488_1599849183566240_7580205458886486901_n.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # make gray image!

#faces = face_cascade.detectMultiScale(gray, 1.3, 5)
faces = catface_cascade.detectMultiScale(gray, scaleFactor=1.04,minNeighbors=2)
print(faces)
# for (x,y,w,h) in faces:
#     image = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
#     roi_gray = gray[y:y+h, x:x+w]
#     roi_color = image[y:y+h, x:x+w]

for (i, (x, y, w, h)) in enumerate(faces):
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(image, "Cat #{}".format(i + 1), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

cv2.imshow('img',image)
cv2.waitKey(0)
cv2.destroyAllWindows()

catDict = {
    'Abyssinian': 198,
    'Bengal': 200,
    'Birman': 200,
    'Bombay': 200,
    'British Shorthair': 184,
    'Egyptian Mau': 200,
    'Main Coon': 190,
    'Persian': 200,
    'Ragdoll': 200,
    'Russian Blue': 200,
    'Siamese': 199,
    'Sphynx': 200
}

for breed, x in catDict.items():
    print(breed)
    for j in range(1,x+1):
        catfile =  'C:/Users/Laura/Documents/TEMP/cats/images/' + str(breed) + '_' + str(j) + '.jpg'
        print(catfile)
        try:
            image = cv2.imread(catfile)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # make gray image!
            faces = catface_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=2)
            for (i, (x, y, w, h)) in enumerate(faces):
                newcatfile = 'C:/Users/Laura/Documents/TEMP/cats/images/catfaces/' + str(breed) + str(j) + '_' + str(i) + '.jpg'
                print('new cat file: ' + newcatfile)
                try:
                    cv2.imwrite(newcatfile, image[y:y+h, x:x+w])
                except:
                    print("oops, not working")
        except Exception:
            print("probably no such file, but might be some other error")
            print(Exception)
            #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            #cv2.putText(image, "Cat #{}".format(i + 1), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)