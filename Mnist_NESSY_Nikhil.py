import sys
import os
sys.path.append(os.getcwd())

import nengo
import numpy as np
from simplified_stdp_Nessy import STDP
from LIFN2S3 import LIF
from InputData import PresentInputWithPause
from Heatmap import AllHeatMapSave,HeatMapSave
from datetime import datetime
from nengo_extras.data import load_mnist
import pickle

np.random.seed(0)
#############################
# load the data
#############################

if(len(sys.argv) > 1):
    NetworkInfo = sys.argv[1]


img_rows, img_cols = 28, 28
input_nbr = 500
Dataset = "Mnist"
(image_train, label_train), (image_test, label_test) = load_mnist()

#select the 0s and 1s as the two classes from MNIST data
image_train_filtered = []
label_train_filtered = []

epoch = 1

for e in range(epoch):
    for i in range(0,input_nbr):
#       if label_train[i] in range(0,2):
            image_train_filtered.append(image_train[i])
            label_train_filtered.append(label_train[i])

print("actual input",len(label_train_filtered))
print(np.bincount(label_train_filtered))


image_train_filtered = np.array(image_train_filtered)
label_train_filtered = np.array(label_train_filtered)

#############################

model = nengo.Network(label="My network",seed=0)

#############################
# Model construction
#############################

if(len(sys.argv) > 1):
    print("loaded")
    # loaded data
    layer1_weights = pickle.load(open( NetworkInfo, "rb" ))[0]
    sim_info = pickle.load(open( NetworkInfo, "rb" ))[1]
    learning_args = pickle.load(open( NetworkInfo, "rb" ))[2]
    neuron_args = pickle.load(open( NetworkInfo, "rb" ))[3]
else:
    sim_info = {
    "presentation_time" : 0.20,
    "pause_time" : 0.15,
    "n_in" : 784,
    "n_neurons" : 20,
    "amplitude" : 0.001,
    "dt" : 0.005 }

    learning_args = {
    "learning_rate":8e-6,
    "alf_p":0.01,#0.01,
    "alf_n":0.009,#0.005,
    "beta_p":1.5,#1.5,src.Log.
    "beta_n":2.5,#2.5,
    "prune":False,
    "stats":True,
    "reinforce":False,
    "STDP_DT":0.005,
    "BatchPerPrune":(sim_info["presentation_time"]+sim_info["pause_time"])*1000 }

    # Neuron Params 
    neuron_args = {
    "spiking_threshold":1,
    "tau_ref":0.01, #0.002
    "inc_n":0.01, #0.01 How much the adaptation state is increased after each spike.
    "tau_rc":0.02, #0.02 how quickly the membrane voltage decays to zero in the absence of input
    "tau_n":1, #1 how quickly the adaptation state decays to zero in the absence of spikes
    "inhib":2 } 

    # weights randomly initiated 
    layer1_weights = np.round(np.random.normal(size=(sim_info["n_neurons"], sim_info["n_in"])),5)

    layer1_weights = np.clip(0.5 + layer1_weights * 0.25,0,1)

# Log
animatedHeatMap = True

with model:

    # input layer 
    picture = nengo.Node(PresentInputWithPause(image_train_filtered, sim_info["presentation_time"],sim_info["pause_time"]),label="Mnist")
    input_layer = nengo.Ensemble(
        sim_info["n_in"],
        1,
        label="Input",
        neuron_type=nengo.LIF(amplitude=0.001),
        encoders=nengo.dists.Choice([[1]]),
        intercepts=nengo.dists.Choice([0]),
        seed=0)

    input_conn = nengo.Connection(picture,input_layer.neurons,synapse=None,seed=0)
    
    # define first layer
    layer1 = nengo.Ensemble(
        sim_info["n_neurons"],
        1,
        label="layer1",
        neuron_type=LIF(spiking_threshold=neuron_args["spiking_threshold"],inc_n=neuron_args["inc_n"],tau_n=neuron_args["tau_n"],tau_rc=neuron_args["tau_rc"], amplitude=0.001),
        gain=nengo.dists.Choice([2]),
        encoders=nengo.dists.Choice([[1]]),
        bias=nengo.dists.Choice([0]),
        seed=0
         )

    conn1 = nengo.Connection(
        input_layer.neurons,
        layer1.neurons,
        transform=layer1_weights, 
        learning_rule_type=STDP(learning_rate=learning_args["learning_rate"],alf_p=learning_args["alf_p"],alf_n=learning_args["alf_n"],beta_p=learning_args["beta_p"],beta_n=learning_args["beta_n"]),seed=0
        )

    #############################
    # setup the probes
    #############################

    layer1_synapses_probe = nengo.Probe(conn1,"weights",label="layer1_synapses") 
    layer1_spikes_probe = nengo.Probe(layer1.neurons,"output",label="layer1_spikes")
    layer1_voltage_probe = nengo.Probe(layer1.neurons,"voltage",label="layer1_voltage")


    #############################

step_time = (sim_info["presentation_time"] + sim_info["pause_time"]) 
Args = {"backend":"Nengo","Dataset":Dataset,"Labels":label_train_filtered,"step_time":step_time,"input_nbr":input_nbr}

with nengo.Simulator(model,dt=sim_info["dt"],progress_bar=True) as sim:
    
    sim.run(step_time * label_train_filtered.shape[0])

now = str(datetime.now().date()) + str(datetime.now().time())
folder = "My_Sim_"+now

if not os.path.exists(folder):
    os.makedirs(folder)

#save the model
pickle.dump([sim.data[layer1_synapses_probe][-1],sim_info,learning_args,neuron_args], open( folder+"/mnist_params_STDP", "wb" ))

for i in range(0,(sim_info["n_neurons"])):
    if(animatedHeatMap):
        AllHeatMapSave(sim,layer1_synapses_probe,folder,sim.data[layer1_synapses_probe].shape[0],i)
    else:
        HeatMapSave(sim,folder,layer1_synapses_probe,sim.data[layer1_synapses_probe].shape[0],i)

print(np.sum(sim.data[layer1_spikes_probe],axis=0)/100)