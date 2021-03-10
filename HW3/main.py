import numpy as np
import matplotlib.pyplot as plt
import math, torch
import argparse, json
from scipy.integrate import odeint
from torch import nn
from torch.autograd import Variable

#Command Line Arguments

parser = argparse.ArgumentParser(description='ODE Solver')
parser.add_argument('--param', metavar='params/param.json',
                    help='parameter file name')
parser.add_argument('-v', type=int, default=1, metavar='N',
                    help='verbosity (default: 1)')
parser.add_argument('--x-field', help='x vector field')
parser.add_argument('--y-field', help='y vector field')
parser.add_argument('--lb', metavar='LB',
                    help='lower bound for initial condition')
parser.add_argument('--ub', metavar='UB',
                    help='upper bound for initial condition')
parser.add_argument('--res-path', metavar='plots',
                    help='path of results')
args, unknown = parser.parse_known_args()

with open(args.param) as paramfile:
        param = json.load(paramfile)

#Converting Strings to Int/Float

lb = float(args.ub)
ub = float(args.lb)
n_p = int(param['data']['n_points'])
x_field = float(args.x-field)
y_field = float(args.y-field)

#Creating Dataset

def rand_points(n_points = n_p, lb, ub):
    return np.random.uniform(lb, ub, n_points)

def create_dataset(fx, fy, dataset_size=param['data']['dataset_size'], look_back=param['exec']['look_back'], epsilon=param['exec']['epsilon']):
    random_x= rand_points(dataset_size)
    random_y= rand_points(dataset_size)
    
    data_in, data_out = [], []
    for x, y in zip(random_x, random_y):
        points= [(x + epsilon*fx(x, y), y + epsilon*fy(x, y))]
        for i in range(look_back):
            x1, y1= points[-1]
            points.append((x1 + epsilon*u(x1, y1), y1 + epsilon*v(x1, y1)))
        data_in.append(points[:-1])
        data_out.append(points[-1])
        
    return np.array(data_in), np.array(data_out)

#Neural Network

class lstm_reg(nn.Module):
    def __init__(self, n_dim, seq_len, n_hidden, n_layers=1):
        super(lstm_reg, self).__init__()
        
        self.rnn = nn.LSTM(n_dim, n_hidden, n_layers, batch_first = True)
        self.fc = nn.Linear(n_hidden * seq_len, n_dim)
        
    def forward(self, x):
        x, _ = self.rnn(x)
        b, s, h = x.shape
        x = self.fc(x)
        return x

seq_len = param['data']['seq_len']
dataset_size = param['data']['dataset_size']
lb, ub = args.lb, args.ub

u = lambda x, y: x_field
v = lambda x, y: y_field  

data_in, data_out = create_dataset(u, v, dataset_size, seq_len, 0.01)
train_in = torch.from_numpy(data_in.reshape(-1, seq_len, 2))
train_out = torch.from_numpy(data_out.reshape(-1, 1, 2))

net = lstm_reg(2, 1, 20)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=param['exec']['learning_rate'])

for e in range(1000):
    var_in = Variable(train_in).to(torch.float32)
    var_out = Variable(train_out).to(torch.float32)
    
    out = net(var_in)
    loss = criterion(out, var_out)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (e + 1)% 100 == 0:
        print(loss)

#Test
        
net = net.eval ()
n_tests= args.n-tests

x, y = np.meshgrid(np.linspace(lb, ub, 10), np.linspace(lb, ub, 10))
plt.quiver(x, y, u(x, y), v(x, y))

for j in range(n_tests):
    x, y= rand_points(), rand_points()
    init_point= torch.from_numpy(np.array((x, y)).reshape(1, 1, 2))
    init_point= Variable(init_point).to(torch.float32)

    all_points= []
    for i in range(dataset_size):
        init_point= net(init_point)
        all_points.append(init_point.detach().numpy())

    all_points= np.array(all_points).reshape(-1, 2)
    plt.plot(*np.array(all_points).T)
    plt.plot(x, y, 'o', markersize= 5, color=plt.gca().lines[-1].get_color())
    plt.savefig(args.res-path + '.pdf')
    
plt.show()

# ODE Solver for Comparison

def ODE_solver(t, z):
    return [u(z[0], z[1]), v(z[0], z[1])]
tspan = [0, 10]

x, y = rand_points(1, lb, ub), rand_points(1, lb, ub)
ode_initial = [x[0], y[0]]
sol = solve_ivp(ODE_solver, tspan, ode_initial, max_step=0.05)

