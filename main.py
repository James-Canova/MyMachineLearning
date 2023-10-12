#main.py
#Micropython and Pyboard V1.1
#Date: 4 March 2022
#version V1
#James Canova
#jscanova@gmail.com

#Based on:
#1)"Make your own Neural Network"
#   https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork/

#Micropython with Pyboard v1.1

#V2:
#update forward propagration and backward propagation equations
#updated learning rate and number of epochs
#much better results now <8% error

import micropython
from pyb import Pin
from ulab import numpy as np
import random as rnd

micropython.alloc_emergency_exception_buf(100)

#--------------------------------------------------------
#Hyperparameters (i.e. they control the solution)
#note: these were selected by trial and error
LEARNING_RATE = 0.04
EPOCHS = 20

#required to obtain consistent results
rnd.seed(30)


#setup--------------------------------------
#identify Pyboard LEDs
red = 1
green = 2
blue = 4
yellow = 3


#setup push button & sliders
pb1 = Pin('X1', Pin.IN, Pin.PULL_UP)
slider1 = Pin('X2', Pin.IN, Pin.PULL_NONE)
slider2 = Pin('X3', Pin.IN, Pin.PULL_NONE)


#setup variables
global bglobal_nQueryPbPushed
bglobal_nQueryPbPushed = False


#neural network------------------------------
#general functions
def sigmoid (x):
    
    return 1/(1 + np.exp(-x))

pass



#--------------------------------------------------------
#for initializing weights and biases between and including:-1, 1 
#shape contains dimensions of required matrix
def create_random_array(shape):

    new_array = np.zeros(shape)

    nRows = shape[0]
    nColumns = shape[1]

    for i in range(nRows):
        for j in range(nColumns):
            new_array[i][j] += rnd.uniform(-1, 1)
    return new_array

#--------------------------------------------------------
#Neural network class
class neuralNetwork:

    # initialise the neural network
    # note: one hidden layer only
    # inputnodes = number of input nodes (i.e neurons or perceptrons)
    # hiddennodes = number of hidden nodes
    # ouptutnodes = numper of output nodes
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate, epochs):

        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.epochs = epochs

        #Initialize weights and bias with random numbers, -1 <= num <= 1 
        self.wih = create_random_array((inputnodes, hiddennodes))   # 2 x 2
        self.who = create_random_array((outputnodes, hiddennodes))  # 1 x 2

        self.bih = create_random_array((hiddennodes,1))  # 2 x 1, added to hidden nodes
        self.bho = create_random_array((outputnodes,1))  # 1 x 1, added to output node

       # learning rate
        self.lr = learningrate

        # number of epochs
        self.epochs = epochs
        
        #flag for training
        self.bTrained = False
        

    # to train, note: targets are the outputs
    # inputs is a 4 x 2 numpy array, each row is a pair of input values
    # targets is a 4 x 1 numpy array
    def train(self, inputs, targets):

        # interate through epochs 
        for c1 in range(self.epochs):

          epoch_cost = 0.0  # error per epoch for plotting cost

          # interate through 4 inputs
          for c2 in range(inputs.shape[0]):  #inputs.shape[0] equals the number of input pairs which is 4

            input = inputs[c2,:]   #input is a 1 x 2 numpy array
            input = input.reshape((2, 1)) # 2 x 1

            target = targets[c2,:]   #target is a 1D numpy array

            #forward propagation-----
            #calculate hidden outputs from inputs
            hidden_sums = np.dot(self.wih, input) + self.bih # 2 x 1 . 1 x 1 + 2 x 1 = 2 x 1 
            hidden_outputs = sigmoid(hidden_sums)  # 2 x 1

            #calculate predicted output from hidden outputs
            output_sum = np.dot(self.who, hidden_outputs) + self.bho # 2 x 2 . 2 x 1 + 2 x 1 = 2 x 1 
            final_output = sigmoid(output_sum)  # 1 x 1
     
        
            #backward propagation-----
            #update weights for hidden to output layer
            output_error = target - final_output   #1 x 1 - 1 x 1 
            dWho = self.lr * np.dot((output_error * final_output * \
                           (1.0 - final_output)), hidden_outputs.T) # 1 x 2
            self.who += dWho

            #update bias for output layer
            dbho = self.lr * output_error * \
                      (final_output * (1.0 - final_output)) # 1 x 1
            self.bho += dbho # 1 x 1


            #update weights for hidden layer
            hidden_error = np.dot(self.who.T, output_error) # 2 x 1 . 1 x 1  [Ref. 2, pp. 79-82 & p. 171]
            dWih = self.lr * np.dot((hidden_error * hidden_outputs * \
                           (1.0 - hidden_outputs)), hidden_sums.T) # 1 x 2
            self.wih += dWih  # 2 x 1

            #update bias(es) for input to hidden layer
            dbih = self.lr * hidden_error * (hidden_outputs * (1.0 - hidden_outputs)) # 2 x 1
            self.bih += dbih  # 2 x 1

        self.bTrained = True

    # to query (i.e. infer results)
    def query(self, input):

      #calculate hidden outputs from inputs
      hidden_sums = np.dot(self.wih, input) + self.bih  # 2 x 2 . 2 x 1 + 2 x 1 = 2 x 1  
      hidden_outputs = sigmoid(hidden_sums)  # 2 x 1

      #calculate predicted output from hidden outputs
      output_sum = np.dot(self.who, hidden_outputs) + self.bho   # 1 x 2 . 2 x 1 + 1 x 1 = 1 x 1 
      inferred_output = sigmoid(output_sum)  # 1 x 1

      return inferred_output # 1 x 1


   # to query (i.e. infer results)
    def query(self, input):

      #calculate hidden outputs from inputs
      hidden_sums = np.dot(self.wih, input) + self.bih  # 2 x 2 . 2 x 1 + 2 x 1 = 2 x 1  
      hidden_outputs = sigmoid(hidden_sums)  # 2 x 1

      #calculate predicted output from hidden outputs
      output_sum = np.dot(self.who, hidden_outputs) + self.bho   # 1 x 2 . 2 x 1 + 1 x 1 = 1 x 1 
      inferred_output = sigmoid(output_sum)  # 1 x 1

      return inferred_output # 1 x 1


#hardware inputs/outputs---------------------
#ISR for push button
#note that pin is type int
def pbChanged(pin):
    
       print("Interupt called")

       global bglobal_nQueryPbPushed
       
       bglobal_nQueryPbPushed = True
       

pass

#set up push button interrupt
extintPb1 = pyb.ExtInt(pb1, pyb.ExtInt.IRQ_FALLING, pyb.Pin.PULL_UP, pbChanged)


#main program--------------------------------
pyb.LED(red).on()      #not yet trained so turn red LED on
pyb.LED(green).off()   #just in case they are still on...
pyb.LED(blue).off()
pyb.LED(yellow).off()

#bglobal_Trained = False

# initialize neural network & then train

# number of input, hidden and output nodes
input_nodes = 2
hidden_nodes = 2
output_nodes = 1

# create instance of neural network
nn = neuralNetwork(input_nodes, hidden_nodes, output_nodes, LEARNING_RATE, EPOCHS)

#--------------------------------------------------------
#train the neural network
#define the training dataset, which are inputs and targets (outputs)
#define inputs and targets for training and query
inputs_array= np.array([[0,0],[0,1],[1,0],[1,1]])
targets_array = np.array([[0],[1],[1],[0]])

print("Training...")
nn.train(inputs_array, targets_array)
print("...training complete.")
bglobal_Trained = nn.bTrained
print(bglobal_Trained) 


pyb.LED(red).off()
pyb.LED(green).on()  #indicates that the neural network is trained
pyb.LED(blue).off()
pyb.LED(yellow).off()


#main loop
while True:
    

    
    if bglobal_nQueryPbPushed == True and nn.bTrained == True:  #then query

        print(bglobal_nQueryPbPushed, nn.bTrained)

        state = machine.disable_irq() #disable all interrupts

        #read state of slider switches
        nValueInput1 = slider1.value()
        nValueInput2 = slider2.value()
        
        inputs_list= numpy.array([[nValueInput1,nValueInput2]])

        final_outputs= nn.query(inputs_list)

        nOutput = int(round(final_outputs[0,0]))
        
        #to account for how the Pyboard is wired internally
        #each input is connected to a 3.3V source with a pull up resistot
        if nOutput == 1:
            nOutput = 0    
        else:
            nOutput = 1
        

        if nOutput == 1: #turn on blue LED

            pyb.LED(blue).on()
            pyb.LED(yellow).off() 

        elif nOutput == 0:  #nOutput is 0 so turn on yellow LED

            pyb.LED(yellow).on()
            pyb.LED(blue).off()

        else:

            pass
 #
        machine.enable_irq(state) #re-enable all interrupts to previous state
        bglobal_nQueryPbPushed = False
        
    pass


    #reset variables to prepare for next query
    global_bPbPushed = False
    bglobal_nQueryPbPushed = False

pass    #end of infinite loop

