import torch
import numpy as np

def generate_data(w, b, num_examples):
    """generate y = X w + b + noise"""
    X = np.random.normal(scale=1, size=(num_examples, len(w)))
    y = np.dot(X, w) + b
    y += np.random.normal(scale=0.01, size=y.shape)
    X=torch.from_numpy(X).float()
    y=torch.from_numpy( y).float().reshape(-1,1)
    return X,y
true_w = torch.Tensor([2, -3.4])
true_b = 4.2
features, labels = generate_data(true_w, true_b, 1000)

from torch.utils.data import TensorDataset,DataLoader

def load_array(data_arrays, batch_size, is_train=True):
    dataset=TensorDataset(*(features,labels))
    dataloader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)
    return dataloader
batch_size =10
data_iter = load_array((features, labels), batch_size)

class LinearRegressionModel(torch.nn.Module): 
    def __init__(self): 
        super(LinearRegressionModel, self).__init__() 
        self.layer1 = torch.nn.Linear(2, 1,bias=True)
    def forward(self, x): 
        y_pred = self.layer1(x)
        return y_pred 
    
net = LinearRegressionModel()
#initialize model params
net.layer1.weight.data=torch.Tensor(np.random.normal(size=(1,2),scale=0.01,loc=0))
net.layer1.bias.data=torch.Tensor([0])

#loss function
loss = torch.nn.MSELoss(reduction = "sum")

#optimization algorithm
trainer = torch.optim.SGD(net.parameters(), lr = 0.03)

#training

num_epochs = 3
for epoch in range(num_epochs): 
    for X,y in data_iter:
        l=loss(net(X) ,y)
        trainer.zero_grad() 
        l.backward() 
        trainer.step() 
    l_epoch = loss(net(features), labels) 
    print('epoch {}, loss {}'.format(epoch+1, l_epoch))


w = list(net.parameters())[0][0]
print('Error in estimating w', true_w.reshape(w.shape) - w)
b = list(net.parameters())[1][0]
print('Error in estimating b', true_b - b)
