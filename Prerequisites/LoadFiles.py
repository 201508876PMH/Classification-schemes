import numpy as np

class LoadFiles():
    def __init__(self):
        pass
                

    def load_ORL_face_data_set_40x30(self):
        content = open("/Users/pmh/Desktop/classification_scheme/Attached_files/ORL_txt/orl_data.txt", 'r')
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

    
    def load_ORL_labels(self):
        content = open("/Users/pmh/Desktop/classification_scheme/Attached_files/ORL_txt/orl_lbls.txt", 'r')
        readContent = content.read().split()
        return readContent