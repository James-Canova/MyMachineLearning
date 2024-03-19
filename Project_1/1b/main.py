#main.py
#Micropython and Rasperry Pi Pico 
#Date created: 19 January 2024
#last updated: 16 March 2024

#James Canova
#jscanova@gmail.com

#Based on:
#1)"Make your own Neural Network"
#   https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork/

#This program solves two inout (one ouput) XOR logic using a neural network

#When ran:
#Red LED is on to indicate that the neural network is not trained
#

import micropython
from machine import Pin
from ulab import numpy as np
import random as rnd
import time

#From Micropython documentaion:
#https://docs.micropython.org/en/v1.9.3/pyboard/library/micropython.html:
#The buffer is used to create exceptions in cases when normal RAM allocation
#would fail (eg within an interrupt handler) and
#therefore give useful traceback information in these situations.
micropython.alloc_emergency_exception_buf(100)

#--------------------------------------------------------
#Hyperparameters (i.e. they control the solution)
#note: these were selected by trial and error
LEARNING_RATE = 0.08
EPOCHS = 20000

#required to obtain consistent results
rnd.seed(30)


#setup LEDs
ledRed = Pin(16, Pin.OUT)
ledGreen = Pin(18, Pin.OUT)
ledBlue = Pin(26, Pin.OUT)
ledYellow = Pin(28, Pin.OUT)


#setup slider switches
SW1 = Pin(14, Pin.IN, Pin.PULL_UP)
SW2 = Pin(15, Pin.IN, Pin.PULL_UP)

#setup LEDs
#red LED on: not trained
#green LED on: trained
#blue LED on: output == 1
#yellow LED on: output == 0


#neural network------------------------------
#general functions
#activation function
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


   # to infere (i.e. infer output from inputs)
    def infere(self, input):

      #calculate hidden outputs from inputs
      hidden_sums = np.dot(self.wih, input) + self.bih  # 2 x 2 . 2 x 1 + 2 x 1 = 2 x 1  
      hidden_outputs = sigmoid(hidden_sums)  # 2 x 1

      #calculate predicted output from hidden outputs
      output_sum = np.dot(self.who, hidden_outputs) + self.bho   # 1 x 2 . 2 x 1 + 1 x 1 = 1 x 1 
      inferred_output = sigmoid(output_sum)  # 1 x 1

      return inferred_output # 1 x 1



#main program--------------------------------
#initialize LEDs
#red LED on: not trained
#green LED on: trained
#blue LED on: output == 1
#yellow LED on: output == 0
ledRed.on()
ledGreen.off() 
ledBlue.off() 
ledYellow.off()    
    

#--------------------------------------------------------
# initialize neural network
# number of input, hidden and output nodes
input_nodes = 2
hidden_nodes = 2
output_nodes = 1

# create instance of neural network
nn = neuralNetwork(input_nodes, hidden_nodes, output_nodes, LEARNING_RATE, EPOCHS)

#--------------------------------------------------------
#train the neural network
#define the training dataset, which are inputs and targets (outputs)
#define inputs and targets for training and infere
inputs_array= np.array([[0,0],[0,1],[1,0],[1,1]])  # 4 x 2
targets_array = np.array([[0],[1],[1],[0]])	# 4 x 1

print("Training...")
nn.train(inputs_array, targets_array)
print("...training complete.")
print("\n")

#indicates that the neural network is trained
pyb.LED(red).off()
pyb.LED(green).on()  


#main loop==================================================================
while True:
    

    #read state of slider switches
    nValueInput0 = slider1.value()
    nValueInput1 = slider2.value()

    print("infereing: {0:.0d}, {1:.0d}".format(nValueInput0, nValueInput1))
    
    #for comparison to inferred result from neural network
    if (nValueInput0 != nValueInput1):
      nExpectedResult = 1;
    else:
      nExpectedResult = 0;        
    
    #set up inputs to neural network
    inputs_list = np.zeros((2,1))
    inputs_list[0,0]= nValueInput0;
    inputs_list[1,0]= nValueInput1;       
    
    #use neural network to infere result (0 or 1)
    final_outputs= nn.infere(inputs_list)
    nInferredResult = int(round(final_outputs[0,0]))
    
    #print inferered result and Expected result
    print("Inferred result:{0}, Expected result:{1}".format(nInferredResult,nExpectedResult))
    print("\n")
    
    time.sleep_ms(500) #to make output readable
    
    #display result using LEDS
    if nInferredResult == 1: #turn on blue LED
        pyb.LED(blue).on()
        pyb.LED(yellow).off()

    elif nInferredResult == 0:  #nOutput is 0 so turn on yellow LED
        pyb.LED(yellow).on()
        pyb.LED(blue).off()          

    else:
        pass

    pass

pass    #end of infinite loop

