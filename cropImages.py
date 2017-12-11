import cv2
import os

newfolder = 'C:/Users/Laura/Documents/TEMP/cats/images/cropped_catfaces/'

i = 0
for file in os.listdir('C:/Users/Laura/Documents/TEMP/cats/images/catfaces'):
    print file
    i = i+1
    newfilename = newfolder + 'cat_' + str(i) + '.jpg'
    filepath = 'C:/Users/Laura/Documents/TEMP/cats/images/catfaces/' + file
    img = cv2.imread(filepath)
    cropped_img = cv2.resize(img, (28,28))
    if i < 10:
        cv2.imshow('image', cropped_img)
        cv2.waitKey(0)
    try:
        cv2.imwrite(newfilename, cropped_img)
    except:
        print("oops, not working")
