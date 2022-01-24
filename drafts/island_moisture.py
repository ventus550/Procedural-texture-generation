from tokenize import Number
import numpy as np

def clearAt(array:np.ndarray,filterValue:Number):
    arrayMax = np.max(array)
    if arrayMax > 1.0:
        array = array/arrayMax
    return np.where(array >= filterValue, array,0)

def joinOnMax(array1,array2):
    com = np.dstack((array1, array2))
    return np.max(com,axis=2)

def normalize(array):
    arrayMax = np.max(array)
    if arrayMax > 1.0:
        array = array/arrayMax
    arrayMin = np.min(array)
    if arrayMin < -1.0:
        array = array/abs(arrayMin)
    return array