import torch
import torch.nn as nn
import torch.nn.functional as func

class Net(nn.Module):

    def __init__(self, n_bits):
        super(Net, self).__init__()
        self.fc1= nn.Linear(n_bits, 100)
        self.fc2= nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 5)

    # Feedforward function
    def forward(self, x):
        h = func.relu(self.fc1(x))
        h2 = func.relu(self.fc2(h))
        y = torch.sigmoid(self.fc2(h2))
        return y

    # Reset function for the training weights
    # Use if the same network is trained multiple times.
    def reset(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()

    # Backpropagation function
    def backprop(self, data, loss, epoch, optimizer):
        self.train()
        inputs= torch.from_numpy(data.x_train)
        targets= torch.from_numpy(data.y_train)
        outputs= self(inputs)
        # An alternative to what you saw in the jupyter notebook is to
        # flatten the output tensor. This way both the targets and the model
        # outputs will become 1-dim tensors.
        print(inputs.shape)
        print(self.forward(inputs).shape)
        print(targets.shape)
        obj_val= loss(self.forward(inputs).reshape(-1), targets)
        optimizer.zero_grad()
        obj_val.backward()
        optimizer.step()
        return obj_val.item()

    # Test function. Avoids calculation of gradients.
    def test(self, data, loss, epoch):
        self.eval()
        with torch.no_grad():
            inputs= torch.from_numpy(data.x_test)
            targets= torch.from_numpy(data.y_test)
            outputs= self(inputs)
            cross_val= loss(self.forward(inputs).reshape(-1), targets)
        return cross_val.item()
