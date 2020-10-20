import d2l
import torch
import torch.nn as nn
import torch.optim as optim
import time

def vgg_block(num_convs, in_channels, out_channels):
    layers=[]
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    
    blk = nn.Sequential(*layers)
    
    return blk

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)
    
def vgg(conv_arch):
    # The convulational layer part
    conv_layers=[]
    in_channels=1
    
    for (num_convs, out_channels) in conv_arch:
        conv_layers.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
    
    net=nn.Sequential(
                      *conv_layers,
                      # The fully connected layer part
                      Flatten(),
                      nn.Linear(in_features=512*7*7, out_features=4096),
                      nn.ReLU(),
                      nn.Dropout(0.5),
                      nn.Linear(4096, 4096),
                      nn.ReLU(),
                      nn.Dropout(0.5),
                      nn.Linear(4096, 10)
                     )
    return net

net = vgg(conv_arch)

X = torch.randn(size=(1,1,224,224), dtype=torch.float32)
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)


lr, num_epochs, batch_size, device = 0.05, 5, 64, d2l.try_gpu()
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)

net.apply(init_weights)
net = net.to(device)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
criterion = nn.CrossEntropyLoss()
d2l.train_ch5(net, train_iter, test_iter, criterion, num_epochs, batch_size, device, lr