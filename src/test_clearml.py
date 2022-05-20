import numpy
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from os import makedirs
from os.path import join
from yaml import safe_load

from tensorflow.keras.metrics import Precision, Recall

from utils.loader import *
from utils.train_utils import *
from utils.models import build_mnetv2

from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import precision_score, recall_score

from clearml import Task

print('imports done')
task = Task.init(project_name='TestProject', task_name='empty task')

print(task)

params = safe_load(open(debug_prefix + "params.yaml"))["train"]

print('next steps in dbug')