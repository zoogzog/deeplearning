import numpy as np
import os
import time
import math
from keras import backend as K


class LossFunction():
    
    def lossSumBinaryCrossEntropy (yTrue, yPred):
    
        EPSILON = K.epsilon()
        
        yPred = K.clip(yPred, EPSILON, 1.0-EPSILON)
        
        out = -(yTrue * K.log(yPred) + (1.0 - yTrue) * K.log(1.0 - yPred))
        
        return K.sum(out, axis=-1)