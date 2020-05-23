import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt

def sigmoid(x, N, k, x0):
    return N / (1 + np.exp(-k * (x - x0)))

def exponenta(x, a , b, c, x0):
    return a * np.power(b, x-x0) + c

def poli(x, a , b, c, d):
    return a * np.power(x, 3) + b * np.power(x, 2) + c * np.power(x, 1) + d

def getError(x, y, func, params):
    N = len(x)
    return 1/N * (np.square(y - func(x, *params)).sum())

functions = [sigmoid, exponenta, poli]

def getDataSet(dataset, label):
    data = dataset[label]
    y = data.to_numpy()
    if y[len(y) -1 ] > 500:
        y0 = np.where(y > 100)[0][0]
        y = y[y0:]
    x = np.array(list(range(0,len(y))))
    return (x,y)

def fitFuntion(dataset, label, verbose = False): 
    x, y = getDataSet(dataset, label)
    bestFitFunc = {}
    minError = -1
    fitting = []
    for f in functions:
        try: 
            popt, pcov = curve_fit(f, x, y)
        except:
            continue
        err = getError(x, y, f, popt)
        ob = {'error': err, 'function': f, 'pcov': pcov, 'popt': popt}
        fitting.append(ob)
        if minError == -1 or err < minError:
            minError = err 
            bestFitFunc = ob
    if verbose:
        for res in fitting:
            print("function: " +  str(res['function'].__name__) + " err: " + str(res['error']) +  " params: ", str(res['popt']))
    return bestFitFunc
def plotResult(dataset, label, bestFitFunc):
    x, y = getDataSet(dataset, label)
    func = bestFitFunc['function']
    popt = bestFitFunc['popt']
    plt.axvline(x=len(x)-days_from_today, ymin=0, ymax=y[len(y)-1], color='r')
    plt.plot(x, y, 'b-', label='data')
    plt.plot(x, func(x, *popt), 'g--', label='estimation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
