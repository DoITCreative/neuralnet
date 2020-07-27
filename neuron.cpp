#include "neuron.h"

Neuron::Neuron()
{
    inputNeurons.clear();
    outputNeurons.clear();
    inputWeights.clear();
}

Neuron::~Neuron()
{
    //Unlink neurons
    for (auto &neuron:inputNeurons)
        neuron->unlinkFrom(this);
    for (auto &neuron:outputNeurons)
        this->unlinkFrom(neuron);
}

/*
 * Connects neuron to another one
 */
void Neuron::linkTo(Neuron* neuron)
{
    //Link output
    outputNeurons.push_back(neuron);

    //Link input
    neuron->inputNeurons.push_back(this);
    neuron->inputWeights.push_back(Utils::getRandomDouble());
}

/*
 * Disconnects neuron from another one
 */
void Neuron::unlinkFrom(Neuron* neuron)
{
    //Unlink output
    int i;
    int foundIndex{-1};
    for (i = 0; i < outputNeurons.size(); ++i)
       if (outputNeurons[i] == neuron) 
       {
            foundIndex = i;
            break;
       }
    if (foundIndex != -1)
        outputNeurons.erase(outputNeurons.begin() + foundIndex);

    //Unlink input
    foundIndex = -1;
    for (i = 0; i < neuron->inputNeurons.size(); ++i)
        if (neuron->inputNeurons[i] == this)
        {
            foundIndex = i;
            break;
        }
    if (foundIndex != -1)
    {
        neuron->inputNeurons.erase(neuron->inputNeurons.begin() + foundIndex);
        neuron->inputWeights.erase(neuron->inputWeights.begin() + foundIndex);
    }
}
