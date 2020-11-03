#!/usr/bin/env python3
import matplotlib.pyplot as plt

readfile = open ("err.txt", "r")
data = readfile.read().split("\n")
data.pop()
data = list(map(lambda x: float(x), data))
plt.plot(data)
plt.ylabel('Total error')
plt.xlabel('Training iterations x1000')
plt.show()
