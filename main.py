import random
import re
from sys import get_coroutine_origin_tracking_depth
from sys import exit
random.seed(101)
import matplotlib.pyplot as plt
import math
import matplotlib.patches as mpatches
#from scipy.linalg import svd
import itertools
import torch
import time
import numpy as np
from tqdm import tqdm
from evaluator import ProxyEvaluator
import collections
import os
from data.data import Data
from parse import parse_args

from torch.utils.data import Dataset, DataLoader
# from collect_log import read_log
import torch.nn.functional as F
from model.base.utils import *

# load model
if __name__ == '__main__':
    args = parse_args()
    seed_torch(args.seed) # set random seed

    exec('from model.'+ args.modeltype + ' import ' + args.modeltype + '_RS') # load the model
    RS = eval(args.modeltype + '_RS(args)')

    # activate the recommender system
    RS.execute() # train and test
