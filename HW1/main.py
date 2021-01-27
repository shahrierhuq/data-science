import numpy as np
from matplotlib import pyplot as plt
import argparse
import json

# Adding flags for command line execution
parser = argparse.ArgumentParser()
parser.add_argument("datain", help="data/filename.in")
parser.add_argument("jsonfile", help="data/filename.json")
args = parser.parse_args()

# Loading json file contents
with open(args.jsonfile) as jf:
    params = json.load(jf)

# Loading x and y
if args.datain == 'data/1.in':
    x0 = np.genfromtxt(args.datain, usecols=(0, 1))
    n = x0.shape[0]
    x_ones = np.ones((n, 1))
    x = np.hstack((x0, x_ones))
    y = np.genfromtxt(args.datain, usecols=(2))

else:
    x0 = np.genfromtxt(args.datain, usecols=(0, 1, 2, 3))
    n = x0.shape[0]
    x_ones = np.ones((n, 1))
    x = np.hstack((x0, x_ones))
    y = np.genfromtxt(args.datain, usecols=(4))
    
# Analytic Solution
def analytic():
    x_t = x.transpose()
    w = np.linalg.inv(x_t.dot(x)).dot(x_t).dot(y)
    return w

w_analytic = analytic()

# Gradient Descent
def gradient_desc(learnRate, numIter):
    w = np.zeros(x.shape[1]) # already transposed
    for i in range(0,numIter):
        y_pred = np.matmul(x, w)
        w = w - learnRate*(np.matmul(x.T, y_pred - y))
    return w

w_grad = gradient_desc(params['learning rate'], params['num iter'])

# parse flag string for output file
filename = args.datain.split('.')
output = filename[0]

with open(filename[0] + '.out', 'w') as out:
    for i in w_analytic:
        out.write(str(f'{i:.4f}') + '\n')
    out.write('\n')
    for i in w_grad:
        out.write(str(f'{i:.4f}') + '\n')

jf.close()
out.close()


