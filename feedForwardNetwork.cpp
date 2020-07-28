#include "feedForwardNetwork.h"

void FeedForwardNetwork::initNet(int numberOfInputNeurons, int numberOfOutputNeurons)
{
    inputLayer.clear();
    hiddenLayer.clear();
    outputLayer.clear();

    for (int i = 0; i < numberOfInputNeurons; ++i)
    {
        Neuron *neuron = new Neuron();
        inputLayer.push_back(neuron);

        Neuron *neuronHidden = new Neuron();
        hiddenLayer.push_back(neuronHidden);
    }

    for (int i = 0; i < numberOfOutputNeurons; ++i)
    {
        Neuron *neuronOutput = new Neuron();
        outputLayer.push_back(neuronOutput);
    }

    Neuron *inputLayerBias = new Neuron();
    inputLayerBias->value = 1.0;
    inputLayerBias->isBias = true;
    inputLayer.push_back(inputLayerBias);

    Neuron *hiddenLayerBias = new Neuron();
    hiddenLayerBias->value = 1.0;
    hiddenLayerBias->isBias = true;
    hiddenLayer.push_back(hiddenLayerBias);

    for (auto &inputNeuron: inputLayer)
        for (auto &hiddenNeuron: hiddenLayer)
            if (!hiddenNeuron->isBias)
                inputNeuron->linkTo(hiddenNeuron);
    for (auto &hiddenNeuron: hiddenLayer)
        for (auto &outputNeuron: outputLayer)
            if (!outputNeuron->isBias)
                hiddenNeuron->linkTo(outputNeuron);
}

void FeedForwardNetwork::feedForward()
{
    for (auto &hiddenNeuron: hiddenLayer)
        if (!hiddenNeuron->isBias)
        {
            double sum = 0;
            for (int i = 0; i < hiddenNeuron->inputNeurons.size(); ++i)
                sum += hiddenNeuron->inputNeurons[i]->value * hiddenNeuron->inputWeights[i];
            hiddenNeuron->value = Utils::sigmoid(sum);
        }
    for (auto &outputNeuron: outputLayer)
    {
        double sum = 0;
        for (int i = 0; i < outputNeuron->inputNeurons.size(); ++i)
            sum += outputNeuron->inputNeurons[i]->value * outputNeuron->inputWeights[i];
        outputNeuron->value = Utils::sigmoid(sum);
    }
}

double FeedForwardNetwork::calculateTotalError(std::vector<double>* expectedOutputs)
{
    if (expectedOutputs->size() != outputLayer.size())
        throw std::invalid_argument("Output layer size differs from expected outputs.");

    double error = 0;
    for (int i = 0; i < outputLayer.size(); ++i)
       error += pow(expectedOutputs->at(i) - outputLayer[i]->value, 2) / 2;
    return error;
}

void FeedForwardNetwork::printNet()
{
    std::cout<<"~Input layer:~"<<std::endl;
    for (auto& neuron: inputLayer)
    {
        if (neuron->isBias)
            std::cout<<"Bias neuron: ";
        std::cout<<neuron->value<<std::endl;
    }
    std::cout<<std::endl;

    std::cout<<"~Hidden layer:~"<<std::endl;
    for (auto& neuron: hiddenLayer)
    {
        if (neuron->isBias)
            std::cout<<"Bias neuron: ";
        std::cout<<neuron->value<<std::endl;
        if (neuron->inputWeights.size() != 0)
        {
            std::cout<<"Weights:"<<std::endl;
            for (auto& weight: neuron->inputWeights)
                std::cout<<weight<<std::endl;
        }
        std::cout<<std::endl;
    }

    std::cout<<"~Output layer:~"<<std::endl;
    for (auto& neuron: outputLayer)
    {
        std::cout<<neuron->value<<std::endl;
        std::cout<<"Weights:"<<std::endl;
        for (auto& weight: neuron->inputWeights)
            std::cout<<weight<<std::endl;
        std::cout<<std::endl;
    }
}

double FeedForwardNetwork::propagateBackwards(std::vector<double>* expectedOutputs, double learningRate)
{
    double error_total = calculateTotalError(expectedOutputs);
    std::vector<double> delta;
    delta.clear();

    for (int outputLayerIterator = 0; outputLayerIterator < outputLayer.size(); ++outputLayerIterator)
    {
        double out = outputLayer.at(outputLayerIterator)->value;
        double target = expectedOutputs->at(outputLayerIterator);
        delta.push_back((out - target) * out * (1 - out));
    }

    //Output layer weights deltas
    std::vector<double> weightOverwriteOutputLayer;
    weightOverwriteOutputLayer.clear();
    for (int outputLayerIterator = 0; outputLayerIterator < outputLayer.size(); ++outputLayerIterator)
    {
        for (int weightsIterator = 0;
             weightsIterator < outputLayer.at(outputLayerIterator)->inputWeights.size();
             ++weightsIterator)
        {
            double currentWeight = outputLayer.at(outputLayerIterator)->inputWeights.at(weightsIterator);
            double deltaError_div_deltaWeight = delta.at(outputLayerIterator) * outputLayer.at(outputLayerIterator)->inputNeurons.at(weightsIterator)->value;
            
            double weightOverwrite = currentWeight - learningRate * deltaError_div_deltaWeight;
            weightOverwriteOutputLayer.push_back(weightOverwrite);
        }
    }

    //Hidden layer weights deltas
    std::vector<double> weightOverwriteHiddenLayer;
    weightOverwriteHiddenLayer.clear();
    for (int hiddenLayerIterator = 0; hiddenLayerIterator < hiddenLayer.size(); ++hiddenLayerIterator)
    {
        if (hiddenLayer.at(hiddenLayerIterator)->isBias)
            continue;

        for (int weightsIterator = 0;
             weightsIterator < hiddenLayer.at(hiddenLayerIterator)->inputWeights.size();
             ++weightsIterator)
        {
            double currentWeight = hiddenLayer.at(hiddenLayerIterator)->inputWeights.at(weightsIterator);
            double deltaError_div_deltaWeight = 0;
            for (int deltaIterator = 0; deltaIterator < delta.size(); ++deltaIterator)
            {
               deltaError_div_deltaWeight += (delta.at(deltaIterator) * outputLayer.at(deltaIterator)->inputWeights.at(0));
            }
            double out_hidden = hiddenLayer.at(hiddenLayerIterator)->value;
            deltaError_div_deltaWeight *= out_hidden * (1 - out_hidden);
            double input_value = hiddenLayer.at(hiddenLayerIterator)->inputNeurons.at(0)->value;
            deltaError_div_deltaWeight *= input_value;
            double weightOverwrite = currentWeight - learningRate * deltaError_div_deltaWeight;
            weightOverwriteHiddenLayer.push_back(weightOverwrite);
        }
    }

    int couter = 0;
    for (int outputLayerIterator = 0; outputLayerIterator < outputLayer.size(); ++outputLayerIterator)
    {
        for (int weightsIterator = 0; 
             weightsIterator < outputLayer.at(outputLayerIterator)->inputWeights.size();
             ++weightsIterator)
        {
            outputLayer.at(outputLayerIterator)->inputWeights.at(weightsIterator) = weightOverwriteOutputLayer.at(couter);
            couter++;
        }
    }

    couter = 0;
    for (int hiddenLayerIterator = 0; hiddenLayerIterator < hiddenLayer.size(); ++hiddenLayerIterator)
    {
        for (int weightsIterator = 0; 
             weightsIterator < hiddenLayer.at(hiddenLayerIterator)->inputWeights.size();
             ++weightsIterator)
        {
            hiddenLayer.at(hiddenLayerIterator)->inputWeights.at(weightsIterator) = weightOverwriteHiddenLayer.at(couter);
            couter++;
        }
    }
    feedForward();
    return calculateTotalError(expectedOutputs);
}
