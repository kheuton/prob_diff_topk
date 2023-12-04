import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon

class ZeroInflatedDist(object):
    
    def __init__(self, dist, zero_proba):
        self.dist = dist
        self.zero_proba = float(zero_proba)
        
    def rvs(self, size=1, random_state=np.random):
        vals = np.atleast_1d(np.round(self.dist.rvs(size=size, random_state=random_state)))
        zmask = random_state.rand(size) < self.zero_proba
        vals[zmask] = 0
        return np.maximum(0, vals)
    
class QuantizedNormal(object):
    
    def __init__(self, loc, scale):
        self.dist = scipy.stats.norm(loc, scale)
    
    def rvs(self, *args, **kwargs):
        vals = np.atleast_1d(np.round(self.dist.rvs(*args, **kwargs)))
        return np.maximum(0, vals)