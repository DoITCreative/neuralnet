#!/usr/bin/env python3
import matplotlib.pyplot as plt

readfile = open ("err.txt", "r")
data = readfile.read().split("\n")
plt.plot(data)
plt.gca().invert_yaxis()
plt.ylabel('Total error')
plt.xlabel('Training iterations x1000')
plt.show()
