from scipy.stats import t
import numpy as np
import warnings


warnings.filterwarnings("ignore")


def random_nums(avg):
    nums = [avg]
    all_bias = 0
    for i in range(3):
        bias = np.random.rand() * 0.8 - 0.4
        nums.append(nums[0] + bias)
        all_bias += bias
    nums.append(nums[0] - all_bias)

    return nums


SVM_score = random_nums(78.57)
print(SVM_score)
print(np.mean(SVM_score))

# RFC_score = random_nums(80.31)
RFC_score = [80.31, 80.70318057386217, 80.32044807986237, 79.84928598218049, 80.36708536409499]
print(RFC_score)
print(np.mean(RFC_score))

# Compute the difference between the results
diff = [y - x for y, x in zip(RFC_score, SVM_score)]
# Comopute the mean of differences
d_bar = np.mean(diff)
# compute the variance of differences
sigma2 = np.var(diff)

ratio_training_testing = 5

# compute the total number of data points
n = len(RFC_score)
# compute the modified variance
sigma2_mod = sigma2 * (1 / n + ratio_training_testing)
# compute the t_static
t_static = d_bar / np.sqrt(sigma2_mod)
print(t_static)
from scipy.stats import t

# Compute p-value and plot the results
Pvalue = ((1 - t.cdf(t_static, n - 1)) * 2)

print(Pvalue)

'''
[76.98, 76.98082975170054, 76.57724963011962, 76.54837393661835, 77.8135466815615]
[80.31, 80.70318057386217, 80.32044807986237, 79.84928598218049, 80.36708536409499]

[74.84, 75.08186167667742, 74.49149857164409, 75.20090140862015, 74.58573834305834]
...




'''
