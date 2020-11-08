import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data

embedding = 4   # Embedding dimension for autoregressive model
T = 1000        # Generate a total of 1000 points
time = torch.arange(0.0,T)
x = torch.sin(0.01 * time) + 0.2*torch.randn(T)
plt.plot(time.numpy(), x.numpy())

features = torch.zeros((T-embedding, embedding))
for i in range(embedding):
    features[:,i] = x[i:T-embedding+i]
labels = x[embedding:]

ntrain = 600
train_data = torch.utils.data.TensorDataset(features[:ntrain,:], labels[:ntrain])
test_data = torch.utils.data.TensorDataset(features[ntrain:,:], labels[ntrain:])


# Function for initializing the weights of net
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)

# Vanilla MLP architecture
def get_net():
    net = nn.Sequential()
    net.add_module('Linear_1', nn.Linear(4, 10, bias = False))
    net.add_module('relu1', nn.ReLU())
    net.add_module('Linear_2', nn.Linear(10, 10, bias = False))
    net.add_module('relu2', nn.ReLU())
    net.add_module('final', nn.Linear(10, 1, bias = False))
    net.apply(init_weights)
    return net

loss = nn.MSELoss()     #L2 loss =  MSELoss in Pytorch

# Simple optimizer using Adam, random shuffle, minibatch size 16
def train_net(net, data, loss, epochs, learningrate):
    batch_size = 16
    trainer = torch.optim.Adam(net.parameters(), lr= learningrate)
    data_iter = torch.utils.data.DataLoader(data, batch_size = batch_size, shuffle=True)
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for X, y in data_iter:
            trainer.zero_grad()
            output = net(X)
            los = loss(output,y.reshape(-1,1))
            los.backward()
            trainer.step()
            running_loss += los.item()
        
        print('epoch %d, loss: %f' % (epoch, running_loss))
    return net

net = get_net()
net = train_net(net, train_data, loss, 10, 0.01)

l = loss(net(test_data[:][0]), test_data[:][1].reshape(-1,1))
print('test loss: %f' % l.mean().detach().numpy())

estimates = net(features)
plt.plot(time.numpy(), x.numpy(), label='data');
plt.plot(time[embedding:].numpy(), estimates.detach().numpy(), label='estimate');
plt.legend();

predictions = torch.zeros_like(estimates)
predictions[:(ntrain-embedding)] = estimates[:(ntrain-embedding)]
for i in range(ntrain-embedding, T-embedding):
    predictions[i] = net(
        predictions[(i-embedding):i].reshape(1,-1)).reshape(1)

plt.plot(time.numpy(), x.numpy(), label='data');
plt.plot(time[embedding:].numpy(), estimates.detach().numpy(), label='estimate');
plt.plot(time[embedding:].numpy(), predictions.detach().numpy(),
         label='multistep');
plt.legend();

