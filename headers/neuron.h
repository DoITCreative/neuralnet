#ifndef NEURON_H 
#define NEURON_H

#include <vector>
#include "utils.h"

class Neuron
{
    public:
        Neuron();
        ~Neuron();
        double value {0};
        std::vector<Neuron*> inputNeurons;
        std::vector<Neuron*> outputNeurons;
        std::vector<double> inputWeights;
        void linkTo(Neuron* neuron);
        void unlinkFrom(Neuron* neuron);
        bool isBias {false};
    private:
};

#endif //NEURON_H 
