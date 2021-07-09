import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(12, 6), dpi=80)

trian_no_negativ_sample = [0.6739130435, 0.6739130435, 0.6956521739]
trian_random_negativ_sample = [0.7173913043, 0.7391304348, 0.731884058]
trian_shift_negativ_sample = [0.6956521739, 0.7608695652, 0.7608695652]
hyper_param = []

x = ["hyper_param1", "hyper_param2", "hyper_param3"]

plt.subplot(x,
            trian_no_negativ_sample,
            label="random negativ sampling without weights")
plt.subplot(x,
            trian_random_negativ_sample,
            color="red",
            label="random negativ sampling")
plt.subplot(x, trian_shift_negativ_sample, label="shift negativ sampling")
plt.legend(loc='upper left')
plt.show()