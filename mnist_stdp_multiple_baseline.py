
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
from args_mnist import args as my_args
import random
from utilis import *

def evaluate_mnist_multiple_baseline(args):

    #############################
    # load the data
    #############################
    input_nbr = args.input_nbr

    presentation_time = args.presentation_time

    x = args.digit
    np.random.seed(args.seed)
    random.seed(args.seed)

    img_rows, img_cols = 28, 28
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

    
    image_test_filtered = []
    label_test_filtered = []


    for i in range(0,10000):
        image_test_filtered.append(image_test[i])
        label_test_filtered.append(label_test[i])

    image_test_filtered = np.array(image_test_filtered)
    label_test_filtered = np.array(label_test_filtered)



    sim_info = {
    "presentation_time" : args.presentation_time,
    "pause_time" : args.pause_time,
    "n_in" : args.n_in,
    "n_neurons" : args.n_neurons,
    "amplitude" : args.amp_neuron,
    "dt" : args.dt }

    learning_args = {
    "learning_rate":8e-6,
    "alf_p":args.alpha_p,#0.01,
    "alf_n":args.alpha_n,#0.009,
    "beta_p":args.beta_p,#1.5,src.Log.
    "beta_n":args.beta_n,#2.5,
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

    n_neurons = args.n_neurons
    layer1_weights = np.round(np.random.normal(size=(sim_info["n_neurons"], sim_info["n_in"])),5)

    layer1_weights = np.clip(0.5 + layer1_weights * 0.25,0,1)

    animatedHeatMap = False


    model = nengo.Network("My network", seed = args.seed)
    #############################
    # Model construction
    #############################
    with model:
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

        # layer1_synapses_probe = nengo.Probe(conn1,"weights",label="layer1_synapses") 
        # layer1_spikes_probe = nengo.Probe(layer1.neurons,"output",label="layer1_spikes")
        # layer1_voltage_probe = nengo.Probe(layer1.neurons,"voltage",label="layer1_voltage")

        layer1_synapses_probe = nengo.Probe(conn1,"weights",label="layer1_synapses", sample_every=5)
        

    step_time = (sim_info["presentation_time"] + sim_info["pause_time"]) 
    Args = {"backend":"Nengo","Dataset":Dataset,"Labels":label_train_filtered,"step_time":step_time,"input_nbr":input_nbr}

    # with nengo_ocl.Simulator(model) as sim :   
    with nengo.Simulator(model, dt=args.dt, optimize=True) as sim:

        sim.run(step_time * label_train_filtered.shape[0])

    last_weight = sim.data[layer1_synapses_probe][-1]

    sim.close()

    pause_time = 0
    #Neuron class assingment

    model = nengo.Network("My network", seed = args.seed)

    with model:

        picture = nengo.Node(nengo.processes.PresentInput(image_train_filtered, sim_info["presentation_time"]))

        true_label = nengo.Node(nengo.processes.PresentInput(label_train_filtered, presentation_time=args.presentation_time))

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
            transform=last_weight)

        #Probes
        p_true_label = nengo.Probe(true_label)
        p_layer_1 = nengo.Probe(layer1.neurons)

    # with nengo_ocl.Simulator(model) as sim :   
    with nengo.Simulator(model, dt=args.dt, optimize=True) as sim:
        
        sim.run((args.presentation_time) * args.input_nbr)
    
    t_data = sim.trange()
    labels = sim.data[p_true_label][:,0]
    output_spikes = sim.data[p_layer_1]
    neuron_class = np.zeros((n_neurons, 1))
    n_classes = 10
    for j in range(n_neurons):
        spike_times_neuron_j = t_data[np.where(output_spikes[:,j] > 0)]
        max_spike_times = 0 
        for i in range(n_classes):
            class_presentation_times_i = t_data[np.where(labels == i)]
            #Normalized number of spikes wrt class presentation time
            num_spikes = len(np.intersect1d(spike_times_neuron_j,class_presentation_times_i))/(len(class_presentation_times_i)+1)
            if(num_spikes>max_spike_times):
                neuron_class[j] = i
                max_spike_times = num_spikes
    spikes_layer1_probe_train = sim.data[p_layer_1]



    #Testing

    images = image_test_filtered
    labels = label_test_filtered



    input_nbr = int(input_nbr/6)
    
    model = nengo.Network(label="My network",)

    with model:

        # picture = nengo.Node(PresentInputWithPause(images, presentation_time, pause_time,0))
        picture = nengo.Node(nengo.processes.PresentInput(images, presentation_time=args.presentation_time))
        true_label = nengo.Node(nengo.processes.PresentInput(labels, presentation_time=args.presentation_time))
        # true_label = nengo.Node(PresentInputWithPause(labels, presentation_time, pause_time,-1))
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
            transform=last_weight)
        p_true_label = nengo.Probe(true_label)
        p_layer_1 = nengo.Probe(layer1.neurons)


    with nengo.Simulator(model,dt=args.dt) as sim:
           
        sim.run(args.presentation_time * input_nbr)

    accuracy_2 = evaluation_v2(10,n_neurons,int(((args.presentation_time * label_test_filtered.shape[0]) / sim.dt) / input_nbr),spikes_layer1_probe_train,label_train_filtered,sim.data[p_layer_1],label_test_filtered,sim.dt)


    labels = sim.data[p_true_label][:,0]
    t_data = sim.trange()
    output_spikes = sim.data[p_layer_1]
    n_classes = 10
    predicted_labels = []  
    true_labels = []
    correct_classified = 0
    wrong_classified = 0

    class_spikes = np.ones((10,1))

    for num in range(input_nbr):
        #np.sum(sim.data[my_spike_probe] > 0, axis=0)

        output_spikes_num = output_spikes[num*int((presentation_time + pause_time) /args.dt):(num+1)*int((presentation_time + pause_time) /args.dt),:] # 0.350/0.005
        num_spikes = np.sum(output_spikes_num > 0, axis=0)

        for i in range(n_classes):
            sum_temp = 0
            count_temp = 0
            for j in range(n_neurons):
                if((neuron_class[j]) == i) : 
                    sum_temp += num_spikes[j]
                    count_temp +=1
        
            if(count_temp==0):
                class_spikes[i] = 0
            else:
                class_spikes[i] = sum_temp
                # class_spikes[i] = sum_temp/count_temp

        # print(class_spikes)
        k = np.argmax(num_spikes)
        # predicted_labels.append(neuron_class[k])
        class_pred = np.argmax(class_spikes)
        predicted_labels.append(class_pred)

        true_class = labels[(num*int((presentation_time + pause_time) /args.dt))]

        if(class_pred == true_class):
            correct_classified+=1
        else:
            wrong_classified+=1

        
    accuracy = correct_classified/ (correct_classified+wrong_classified)*100
    print("Accuracy: ", accuracy)
    sim.close()

    del sim.data, labels, class_pred, spikes_layer1_probe_train

    return accuracy, accuracy_2, last_weight


    # for tstep in np.arange(0, len(weights), 1):
    #     tstep = int(tstep)
    #     print(tstep)
    #     fig, axes = plt.subplots(1,1, figsize=(3,3))

    #     for i in range(0,(n_neurons)):
            
    #         fig = plt.figure()
    #         ax1 = fig.add_subplot()
    #         cax = ax1.matshow(np.reshape(weights[tstep][i],(28,28)),interpolation='nearest', vmax=1, vmin=0)
    #         fig.colorbar(cax)

    #     plt.tight_layout()    
    #     fig.savefig(folder+'/weights'+str(tstep)+'.png')
    #     plt.close('all')

    # gen_video(folder, "weights")

    # for tstep in np.arange(0, len(weights), 1):
    #     tstep = int(tstep)
    #     print(tstep)
    #     fig, axes = plt.subplots(1,1, figsize=(3,3))

    #     for i in range(0,(n_neurons)):
            
    #         fig = plt.figure()
    #         ax1 = fig.add_subplot()
    #         cax = ax1.hist(weights[tstep][i])
    #         ax1.set_xlim(0,1)
    #         ax1.set_ylim(0,350)

    #     plt.tight_layout()    
    #     fig.savefig(folder+'/histogram'+str(tstep)+'.png')
    #     plt.close('all')

    # gen_video(folder, "histogram")



if __name__ == '__main__':
    logger = logging.getLogger(__name__)

    args = my_args()


    print(args.__dict__)
    logging.basicConfig(level=logging.DEBUG)
    # Fix the seed of all random number generator
    seed = 500
    random.seed(seed)
    np.random.seed(seed)



    # params = nni.get_next_parameter()

    # args.g_max = params['g_max']
    # args.tau_in = params['tau_in']
    # args.tau_out = params['tau_out']
    # args.lr = params['lr']
    # args.presentation_time = params['presentation_time']
    # args.rate_out = params['rate_out']



    accuracy, weights = evaluate_mnist_multiple(args)
    print('accuracy:', accuracy)

    # now = time.strftime("%Y%m%d-%H%M%S")
    # folder = os.getcwd()+"/MNIST_VDSP"+now
    # os.mkdir(folder)


    # plt.figure(figsize=(12,10))

    # plt.subplot(2, 1, 1)
    # plt.title('Input neurons')
    # rasterplot(time_points, p_input_layer)
    # plt.xlabel("Time [s]")
    # plt.ylabel("Neuron index")

    # plt.subplot(2, 1, 2)
    # plt.title('Output neurons')
    # rasterplot(time_points, p_layer_1)
    # plt.xlabel("Time [s]")
    # plt.ylabel("Neuron index")

    # plt.tight_layout()

    # plt.savefig(folder+'/raster'+'.png')


    # for tstep in np.arange(0, len(weights), 1):
    #     tstep = int(tstep)
    #     # tstep = len(weightds) - tstep -1


    #     print(tstep)

    #     columns = int(args.n_neurons/5)
    #     fig, axes = plt.subplots(int(args.n_neurons/columns), int(columns), figsize=(20,25))

    #     for i in range(0,(args.n_neurons)):

    #         axes[int(i/columns)][int(i%columns)].matshow(np.reshape(weights[tstep][i],(28,28)),interpolation='nearest', vmax=1, vmin=0)


    #     plt.tight_layout()    
    #     fig.savefig(folder+'/weights'+str(tstep)+'.png')
    #     plt.close('all')

    # gen_video(folder, "weights")



    logger.info('All done.')