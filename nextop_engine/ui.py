import os
import sys
path_name= os.path.dirname(__file__)
sys.path.append(path_name)

from _element.feature_control import *
from _element.calculations import *
from _usecase.algorithm import *

def do_predict():
    result = []
    accuracy = 100
    return result, accuracy



def do_something_with_usecase():
    return 0
