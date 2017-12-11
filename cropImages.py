import cv2
import os

newfolder = 'C:/Users/Laura/Documents/MASTER/Computational Cognitive Neuroscience/Final Project/cats/Cat/cropped_catfaces/'

for file in os.listdir('C:/Users/Laura/Documents/MASTER/Computational Cognitive Neuroscience/Final Project/cats/Cat/catfaces'):
    print file
    newfilename = newfolder + file # new name is same as old name, so they can be compared!
    filepath = 'C:/Users/Laura/Documents/MASTER/Computational Cognitive Neuroscience/Final Project/cats/Cat/catfaces/' + file
    img = cv2.imread(filepath)
    cropped_img = cv2.resize(img, (28,28))
    try:
        cv2.imwrite(newfilename, cropped_img)
    except:
        print("oops, not working")
