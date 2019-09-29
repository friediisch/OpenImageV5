from fastai.vision import open_image
import pickle
from PIL import Image as pil_im
import warnings
from cv2 import imread
import os 

def SaveAndLoad(filename, data=None):
    if filename in os.listdir('data'):
        with open('data/' + filename, 'rb') as f:
            return pickle.load(f)
    else:
        with open('data/' + filename, 'wb') as f:
            pickle.dump(data, f)  
        

def save(name, data):
    with open('data/' + name, 'wb') as f:
        pickle.dump(data, f)    
def load(name):
    with open('data/' + name, 'rb') as f:
        return pickle.load(f)
    
def get_XY(directory):
    """ Get the shape of all images in target directory"""
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        X = {}
        Y = {}
        for filepath in list(directory.iterdir()):
            try:
                X[filepath.stem], Y[filepath.stem] = pil_im.open(filepath).size
            except:
                print("Image {} corrupted, using cv2.imread()".format(filepath))
                Y[filepath.stem], X[filepath.stem] = imread(str(filepath)).shape[:-1]
    return X,Y

def isListEmpty(inList):
    if isinstance(inList, list): # Is a list
        return all( map(isListEmpty, inList) )
    return False