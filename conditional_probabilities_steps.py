######
# Let's see how Naive bayes works!
# 베이지안 직접 구축해보기!
# 그래프 그려보고 좌표 직접 찍어서 조건부 확률이 어떻게 계산되는지 알아보자
####

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy

#### training sample definition ####
# let's say there are three attributes = x1, x2, x3
# and there are 5 samples for each attribute
# there are two classes, normal and abnormal ==> binary classification

########## definition of data! #################

## normal data
x1 = [0.9, 0.8, 1.2, 1.0, 0.6]
x2 = [0.16, 0.14, 0.18, 0.13, 0.2]
x3 = [0.1, 0.09, 0.11, 0.12, 0.11]

## abnormal data
ax1 = [1.8, 0.35, 1.7, 0.4, 1.5]
ax2 = [0.3, 0.08, 0.09, 0.25, 0.28]
ax3 = [0.18, 0.05, 0.17, 0.06, 0.04]


# 5 samples with three attributes for test
sample1 = [1.05, 0.165, 0.95]
sample2 = [0.5, 0.3, 0.07]
sample3 = [0.85, 0.17, 0.14]
sample4 = [0.45, 0.085, 0.096]
sample5 = [0.7, 0.28, 0.085]

# organize test data according to attributes
test_x1 = [1.05, 0.5, 0.85, 0.45, 0.7]
test_x2 = [0.165, 0.3, 0.17, 0.085, 0.28]
test_x3 = [0.95, 0.07, 0.14, 0.096, 0.085]



# define mean and std function to draw gaussian graph based on mean and std!
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


# calculate conditional probability!
# ex) P(A|B)
def cond_p(x, norm, abnorm):
    norm_rv = scipy.stats.norm(loc=mean(norm), scale=std(norm))
    abnorm_rv = scipy.stats.norm(loc=mean(abnorm), scale=std(abnorm))

    # .pdf(x) => y point of normal gaussian distribution, calcualte P(A|B)!
    return norm_rv.pdf(x), abnorm_rv.pdf(x)


# normal mean and std list
n_mean_list = [mean(x1), mean(x2), mean(x3)]
n_std_list = [std(x1), std(x2), std(x3)]

# abnormal mean and std list
mean_list = [mean(ax1), mean(ax2), mean(ax3)]
std_list = [std(ax1), std(ax2), std(ax3)]

# conditional probability for each class(normal, abnormal)
pnorm = []
pabnorm = []
pnorm2 = []
pabnorm2 = []
pnorm3 = []
pabnorm3 = []

## calculate conditional probabilities of each attributes!

## pnorm = P(X1|C1), pabnorm = P(X1|C2)
print("++++++++++ test1 +++++++++++++++++++")
for i in range(5):
    print(f"conditional probability {i + 1} :", cond_p(test_x1[i] , x1, ax1))
    n, an = cond_p(test_x1[i], x1, ax1)
    pnorm.append(n)
    pabnorm.append(an)

## pnorm = P(X2|C1), pabnorm = P(X2|C2)
print("++++++++++ test2 +++++++++++++++++++")
for i in range(5):
    print(f"conditional probability {i + 1} :", cond_p(test_x2[i] , x2, ax2))
    n, an = cond_p(test_x2[i], x2, ax2)
    pnorm2.append(n)
    pabnorm2.append(an)

## pnorm = P(X3|C1), pabnorm = P(X3|C2)
print("++++++++++ test3 +++++++++++++++++++")
for i in range(5):
    print(f"conditional probability {i + 1} :", cond_p(test_x3[i] , x3, ax3))
    n, an = cond_p(test_x3[i], x3, ax3)
    pnorm3.append(n)
    pabnorm3.append(an)

# print to check
for i in range(5):
    print(i ,": p(x|c1)", pnorm[i] * pnorm2[i] * pnorm3[i])
    print(i ,": p(x|c2)", pabnorm[i] * pabnorm2[i] * pabnorm3[i])



######## draw graphs  #########
legend = []
x = np.arange(0.0 , 2, 0.005)
# attribute x1 graph
legend.append(f'normal x1 = N({n_mean_list[0]},{n_std_list[0]})')
legend.append(f'abnormal x1 = N({mean_list[0]},{std_list[0]})')
plt.fill_between(x, norm.pdf(x,mean_list[0], std_list[0]), alpha=0.5)
plt.fill_between(x, norm.pdf(x,n_mean_list[0], n_std_list[0]), alpha=0.5)


plt.xlabel('x')
plt.ylabel('density')
plt.scatter(test_x1[0], pnorm[0])
plt.scatter(test_x1[0], pabnorm[0])
plt.legend(legend)
plt.savefig("x1_graph.png", dpi=72, bbox_inches='tight')
plt.show()



'''

legend.append(f'abnormal x{i+1} = N({mean_list[0]},{std_list[0]})')
plt.fill_between(x, norm.pdf(x, mean_list[0], std_list[0]), alpha=0.5)
legend.append(f'normal x{i+1} = N({n_mean_list[0]},{n_std_list[0]})')
plt.fill_between(x, norm.pdf(x, n_mean_list[0], n_std_list[0]), alpha=0.5)

plt.xlabel('x')
plt.ylabel('density')
plt.scatter(test_x1[0], pnorm[0])
plt.scatter(test_x1[0], pabnorm[0])
plt.scatter(test_x1[1], pnorm[1])
plt.scatter(test_x1[1], pabnorm[1])
plt.scatter(test_x1[2], pnorm[2])
plt.scatter(test_x1[2], pabnorm[2])

plt.legend(legend)
plt.show()

# normal

print("++++++++++++++++++++++++++++++++++++++")


for i in range(5):
    print(norm.pdf(x2[i],mean(x2), std(x2)))

print("++++++++++++++++++++++++++++++++++++++")


for i in range(5):
    print(norm.pdf(x3[i],mean(x3), std(x3)))

for i in range(len(mean_list)):
    legend.append(f'x{i+1} = N({n_mean_list[i]},{n_std_list[i]})')
    plt.fill_between(x, norm.pdf(x,n_mean_list[i], n_std_list[i]), alpha=0.5)


plt.xlabel('x')
plt.ylabel('density')
plt.legend(legend)
plt.savefig("normal_graph.png", dpi=72, bbox_inches='tight')
plt.show()

#abnormal
for i in range(len(mean_list)):
    legend.append(f'x{i+1} = N({mean_list[i]},{std_list[i]})')
    plt.fill_between(x, norm.pdf(x,mean_list[i], std_list[i]), alpha=0.5)


plt.xlabel('x')
plt.ylabel('density')
plt.legend(legend)
plt.savefig("abnormal_graph.png", dpi=72, bbox_inches='tight')
plt.show()

'''
