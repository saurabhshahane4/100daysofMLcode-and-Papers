import d2l
import torch
import torch.nn as nn

def nin_block(in_channels,out_channels,kernel_size,strides,padding):
    blk = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,strides,padding),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels,kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels,kernel_size=1),
            nn.ReLU())
    return blk

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.n1 = nin_block(1,out_channels=96, kernel_size=11, strides=4, padding=0)
        self.m1 = nn.MaxPool2d(3,stride=2)
        self.n2 = nin_block(96,out_channels=256, kernel_size=5, strides=1, padding=2)
        self.m2 = nn.MaxPool2d(3,stride=2)
        self.n3 = nin_block(256,out_channels=384, kernel_size=3, strides=1, padding=1)
        self.m3 = nn.MaxPool2d(3,stride=2)
        self.dropout1 = nn.Dropout2d(0.5)
        self.n4 = nin_block(384,out_channels=10, kernel_size=3, strides=1, padding=1)
        #Global Average Pooling can be achieved by AdaptiveMaxPool2d with output size = (1,1)
        self.avg1 = nn.AdaptiveMaxPool2d((1,1))
        self.flat = Flatten()
        
    def forward(self, x): 
        x = self.m1(self.n1(x))
        x = self.m2(self.n2(x))
        x = self.dropout1(self.m3(self.n3(x)))
        x = self.n4(x)
        x = self.avg1(x)
        x = self.flat(x) 
        return x

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
net = Net()
X = torch.rand(size=(1,1,224,224))
for layer in net.children():
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)

lr, num_epochs, batch_size, device = 0.1, 5, 128, d2l.try_gpu()

#Xavier initialization of weights
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)

net.apply(init_weights)

#Loading fashion-MNIST data
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

#criterion
criterion = nn.CrossEntropyLoss()

d2l.train_ch5(net, train_iter, test_iter, criterion, num_epochs, batch_size, device, lr)
