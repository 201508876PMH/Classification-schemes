#!/usr/bin/env python3
import numpy as np
from PIL import Image


# Column = (vertical) =  1 colomn = 1200
# Rows = (horizontal) = 1 row = 400

def load_ORL_face_data_set_40x30():

    content = open("../Attached_files/ORL_txt/orl_data.txt", 'r')
    readContent = content.read().split()
    
    matrix = np.zeros(shape=(1200,400))

    counterRow = 0
    counterColumn = 0
    for elem in readContent:
        if(counterColumn == 400):
            counterColumn = 0
            counterRow = counterRow + 1

        matrix[counterRow][counterColumn] = elem
        counterColumn = counterColumn + 1

    return matrix

def fetch_specific_image_in_binary(imageNumber):
    matrix = load_ORL_face_data_set_40x30()
    return matrix[:,imageNumber]

def display_image(imageNumber):

    data = fetch_specific_image_in_binary(imageNumber)
    print(data)
    matrix = np.zeros(shape=(40,30,3), dtype=np.uint8)

    counterRow = 0
    counterColumn = 0
    for elem in data:
        if(counterRow == 40):
            counterRow = 0
            counterColumn = counterColumn + 1

        matrix[counterRow][counterColumn] = elem*255
        counterRow = counterRow + 1

    img = Image.fromarray(matrix, 'RGB')  
    img.save('my.png')
    img.show()

if __name__ == "__main__":
    # print(fetch_specific_image_in_binary(1))
    display_image(10)