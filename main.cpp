#include <iostream>
#include <string>
#include <cmath>

#include "feedForwardNetwork.h"

int main(int argc, char* argv[])
{
    FeedForwardNetwork *n = new FeedForwardNetwork();
    n->initNet(2);
    n->inputLayer.at(0)->value = 0.01;
    n->inputLayer.at(1)->value = 0.01;
    n->feedForward();
    n->printNet();

    std::cout<<"Network initialized"<<std::endl;
    std::cout<<std::endl;

    double error = 0;
    std::cout<<"Learning in progress..."<<std::endl;

    //Learning XOR operation 2-nd output not used
    for (int i = 0; i < 1000000; ++i)
    {
        if (i % 1000 == 0)
            std::cerr<<error<<std::endl;
        n->inputLayer.at(0)->value = 0.00;
        n->inputLayer.at(1)->value = 0.00;
        std::vector<double> expectedOutputs {0.00, 0.00};
        error = n->propagateBackwards(&expectedOutputs, 0.5);

        n->inputLayer.at(0)->value = 0.00;
        n->inputLayer.at(1)->value = 1.00;
        expectedOutputs = {1.00, 0.00};
        error = n->propagateBackwards(&expectedOutputs, 0.5);

        n->inputLayer.at(0)->value = 1.00;
        n->inputLayer.at(1)->value = 0.00;
        expectedOutputs = {1.00, 0.00};
        error = n->propagateBackwards(&expectedOutputs, 0.5);

        n->inputLayer.at(0)->value = 1.00;
        n->inputLayer.at(1)->value = 1.00;
        expectedOutputs = {0.00, 0.00};
        error = n->propagateBackwards(&expectedOutputs, 0.5);
    }
    std::cout<<"Total error value: "<<error<<std::endl;
    std::cout<<std::endl;
    n->printNet();

    delete n;
    return 0;
}
