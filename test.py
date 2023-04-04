
import numpy as np

def f(x, y, averages):
    print("関数内プリント始め")
    averages = np.append(averages, x*y)
    print(averages)
    print("関数内プリント終わり")
    return averages


averages = np.array([], dtype='float')
print(averages)
averages = np.append(averages, 1*1)
print(averages)

averages = f(1,2,averages)
print(averages)
