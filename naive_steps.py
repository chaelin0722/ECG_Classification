import math
import numpy as np
import matplotlib.pyplot as plt
## normal data
x1 = [0.9, 0.8, 1.2, 1.0, 0.6]
x2 = [0.16, 0.14, 0.18, 0.13, 0.2]
x3 = [0.1, 0.09, 0.11, 0.12, 0.11]

## abnormal data
ax1 = [1.8, 0.35, 1.7, 0.4, 1.5]
ax2 = [0.3, 0.08, 0.09, 0.25, 0.28]
ax3 = [0.18, 0.05, 0.17, 0.06, 0.04]

def mean(list):
    m = sum(list) / len(list)

    return m

def std(list):
    v = np.var(list)
    std = math.sqrt(v)

    return std

def draw_plot(mu, sigma):
    s = np.random.normal(mu, sigma, 1000)
    count, bins, ignored = plt.hist(s, 30, density=True)
    plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *
             np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)),
             linewidth=2, color='r')
    plt.show()


print(mean(ax1))
print(mean(ax2))
print(mean(ax3))

print(std(ax1))
print(std(ax2))
print(std(ax3))

draw_plot(0.9, 0.1999)








