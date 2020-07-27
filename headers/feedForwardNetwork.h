#ifndef FEEDFORWARDNETWORK_H 
#define FEEDFORWARDNETWORK_H

#include <vector>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include "neuron.h"

/*
 * Network with 3 layers: 1-st input, 2-nd hidden, 3-rd output
 * Input layer has same number of neurons as hidden layer
 * Each neuron connects to each neuron of next layer
 * Network has no loops
 * Trains with backpropagation algorithm
 * Hidden and output layer include bias neuron
 */
class FeedForwardNetwork
{
    public:
        void initNet(int numberOfInputNeurons);
        std::vector<Neuron*> inputLayer;
        std::vector<Neuron*> hiddenLayer;
        std::vector<Neuron*> outputLayer;
        double learningRate {0.5};
        void feedForward();
        double calculateTotalError(std::vector<double>* expectedOutputs);
        void printNet();
        double propagateBackwards(std::vector<double>* expectedOutputs, double learningRate);
    private:
};

#endif //FEEDFORWARDNETWORK_H 
