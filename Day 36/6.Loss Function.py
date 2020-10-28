loss_func = nn.NLLLoss(reduction='sum')

torch.manual_seed(0)
n,c=8,2
y = torch.randn(n,c, requires_grad=True)
ls_F = nn.LogSoftmax(dim=1)
y_out=ls_F(y)
print(y_out.shape)

target = torch.randint(c, soze=(n,))
print(target.shape)

loss = loss_func(y_out, target)
print(loss.item())

loss.backward()
print(y.data)

from torch import optim
opt - optim.Adam(cnn_model.parameters(), lr=3e-4)
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

current_lr = get_lr(opt)
print('current lr={}'.format(current_lr))

from torch.optim.lr_scheduler import ReduceLROnPlateau
lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5,patience=20,verbose=1)

for i in range(100):
    lr_scheduler.step(1)