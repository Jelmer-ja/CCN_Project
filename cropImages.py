import cv2
import os

newfolder = 'C:/Users/Laura/Documents/MASTER/Computational Cognitive Neuroscience/Final Project/cats/Cat/large_cropped_catfaces/'

curFolder = 'C:/Users/Laura/Documents/MASTER/Computational Cognitive Neuroscience/Final Project/cats/Cat/catfaces'

for file in os.listdir(curFolder):
    print file
    newfilename = newfolder + file # new name is same as old name, so they can be compared!
    filepath = curFolder + '/' + file
    img = cv2.imread(filepath)
    cropped_img = cv2.resize(img, (100, 100))
    try:
        cv2.imwrite(newfilename, cropped_img)
    except:
        print("oops, not working")
