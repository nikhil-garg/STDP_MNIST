import nengo
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# import tensorflow as tf
import os
from nengo.dists import Choice
from datetime import datetime
import pickle
from nengo.utils.matplotlib import rasterplot

plt.rcParams.update({'figure.max_open_warning': 0})
import time

from InputData import PresentInputWithPause
# from custom_rule import CustomRule
# from custom_rule import CustomRule_prev

# import nengo_ocl

from nengo.neurons import LIFRate
# from custom_rule import CustomRule
# from custom_rule import CustomRule_prev
from nengo.params import Parameter, NumberParam, FrozenObject
from nengo.dists import Choice, Distribution, get_samples, Uniform

from nengo.utils.numpy import clip, is_array_like

from nengo.connection import LearningRule
from nengo.builder import Builder, Operator, Signal
from nengo.builder.neurons import SimNeurons
from nengo.learning_rules import LearningRuleType
from nengo.builder.learning_rules import get_pre_ens,get_post_ens
from nengo.neurons import AdaptiveLIF
from nengo.synapses import Lowpass, SynapseParam
from nengo.params import (NumberParam,Default)
from nengo.dists import Choice
from nengo.utils.numpy import clip
import numpy as np
import random
import math




def evaluation(classes,n_neurons,presentation_time,spikes_layer1_probe,label_test_filtered,dt):
    
    ConfMatrix = np.zeros((classes,n_neurons))
    labels = np.zeros(n_neurons)
    accuracy = np.zeros(n_neurons)
    total = 0
    Good = 0
    Bad = 0
    # confusion matrix
    x = 0
    for i in label_test_filtered:
            tmp = spikes_layer1_probe[(x*presentation_time):(x+1)*presentation_time].sum(axis=0)
            tmp[tmp < np.max(tmp)] = 0
            tmp[tmp != 0] = 1
            
            ConfMatrix[i] = ConfMatrix[i] + tmp

            x = x + 1
            

    Classes = dict()
    for i in range(0,n_neurons):
        Classes[i] = np.argmax(ConfMatrix[:,i])
    
    x = 0
    for i in label_test_filtered:
        correct = False
        tmp = spikes_layer1_probe[(x*presentation_time):(x+1)*presentation_time].sum(axis=0)
        tmp[tmp < np.max(tmp)] = 0
        tmp[tmp != 0] = 1

        for index,l in enumerate(tmp):
            if(l == 1):
                correct = correct or (Classes[index] == i)
        if(correct):
            Good += 1
        else:
            Bad += 1
        x = x + 1
        total += 1

    return Classes, round((Good * 100)/(Good+Bad),2)


def evaluation_v2(classes,n_neurons,presentation_time,spikes_layer1_probe_train,label_train_filtered,spikes_layer1_probe_test,label_test_filtered,dt):
    
    ConfMatrix = np.zeros((classes,n_neurons))
    labels = np.zeros(n_neurons)
    accuracy = np.zeros(n_neurons)
    total = 0
    Good = 0
    Bad = 0
    # confusion matrix
    x = 0
    for i in label_train_filtered:
            tmp = spikes_layer1_probe_train[(x*presentation_time):(x+1)*presentation_time].sum(axis=0)
            tmp[tmp < np.max(tmp)] = 0
            tmp[tmp != 0] = 1
            
            ConfMatrix[i] = ConfMatrix[i] + tmp

            x = x + 1
            

    Classes = dict()
    for i in range(0,n_neurons):
        Classes[i] = np.argmax(ConfMatrix[:,i])
    
    x = 0
    for i in label_test_filtered:
        correct = False
        tmp = spikes_layer1_probe_test[(x*presentation_time):(x+1)*presentation_time].sum(axis=0)
        tmp[tmp < np.max(tmp)] = 0
        tmp[tmp != 0] = 1

        for index,l in enumerate(tmp):
            if(l == 1):
                correct = correct or (Classes[index] == i)
        if(correct):
            Good += 1
        else:
            Bad += 1
        x = x + 1
        total += 1

    return round((Good * 100)/(Good+Bad),2)

