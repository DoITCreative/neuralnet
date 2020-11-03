#include <iostream>
#include <string>
#include <cmath>

#include "feedForwardNetwork.h"

int main(int argc, char* argv[])
{
    FeedForwardNetwork *n = new FeedForwardNetwork();
    n->initNet(2, 1);
    n->inputLayer.at(0)->value = 0.01;
    n->inputLayer.at(1)->value = 0.01;
    n->feedForward();
    n->printNet();

    std::cout<<"Network initialized"<<std::endl;
    std::cout<<std::endl;

    double error = 1;
    std::cout<<"Learning in progress..."<<std::endl;

    //Learning XOR operation 2-nd output not used
    int i = 0;
    while (error > 0.01 || i < 50000) 
    {
        n->inputLayer.at(0)->value = 0.00;
        n->inputLayer.at(1)->value = 0.00;
        std::vector<double> expectedOutputs {0.00};
        error = n->propagateBackwards(&expectedOutputs, 0.3);

        n->inputLayer.at(0)->value = 0.00;
        n->inputLayer.at(1)->value = 1.00;
        expectedOutputs = {1.00};
        error = n->propagateBackwards(&expectedOutputs, 0.3);

        n->inputLayer.at(0)->value = 1.00;
        n->inputLayer.at(1)->value = 0.00;
        expectedOutputs = {1.00};
        error = n->propagateBackwards(&expectedOutputs, 0.3);

        n->inputLayer.at(0)->value = 1.00;
        n->inputLayer.at(1)->value = 1.00;
        expectedOutputs = {0.00};
        error = n->propagateBackwards(&expectedOutputs, 0.3);

        if (i % 1000 == 0)
            std::cerr<<error<<std::endl;

        i++;
    }

    std::cout<<"Learning took "<<i<<" iterations"<<std::endl;
    std::cout<<"Total error value: "<<error<<std::endl;
    std::cout<<std::endl;
    n->printNet();

    delete n;
    return 0;
}
