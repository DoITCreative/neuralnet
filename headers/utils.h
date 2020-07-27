#ifndef UTILS_H 
#define UTILS_H

#include <random>

class Utils
{
    public:
        static double getRandomDouble()
        {
            std::random_device rd;
            std::mt19937 e2(rd());
            std::uniform_real_distribution<> dist(0, 1);
            return dist(e2);
        }

        static double sigmoid(double x)
        {
            return static_cast<double>(1 / (1 + pow(2.71828, -1 * x)));
        }
    private:
};

#endif //UTILS_H 
