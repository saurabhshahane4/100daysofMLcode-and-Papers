def metrics_batch(output, target):
    pred = output.argmax(dim=1, keepdim=True)
    corrects=pred.eq(target.view_as(pred)).sum().item()
    return corrects

def loss_batch(loss_func, output, target, opt=None):
    loss=loss_func(output, target)
    with torch.no_grad():
        metric_b = metrics_batch(output,target)
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item(), metric_b

def loss_epoch(model,loss_func,dataset_dl,sanity_check=False,opt=None):   
    running_loss=0.0
    running_metric=0.0
    len_data = len(dataset_dl.dataset)

    for xb, yb in dataset_dl:
        xb=xb.type(torch.float).to(device)
        yb=yb.to(device)
        yb_h=model(xb)
        loss_b, metric_b = loss_batch(loss_func, xb, yb, yb_h, opt)
        running_loss+=loss_b
        if metric_b is not None:
            running_metric+=metric_b
        
        if sanity_check is True:
            break
    
    loss=running_loss/float(len_data)
    metric = running_metric/float(len_data)
    return loss, metric

def train_val(model,params):
    num_epochs = params["num_epochs"]
    loss_func=params["loss_func"]
    opt=params["optimizer"]
    train_dl = params["train_dl"]
    val_dl=params["val_dl"]
    sanity_check=params["sanity_check"]
    lr_scheduler=params['lr_scheduler']
    path2weights=params["path2weights"]

loss_history={
    "train": [],
    "val": [],
}

best_model_wts = copy.deepcopy(model.state_dict())
best_loss= float('inf')

for epoch in range(num_epochs):
    current_lr = get_lr(opt)
    print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs- 1, current_lr))
    model.train()
    train_loss,train_metric=loss_epoch(model,loss_func,train_dl,sanity_check,opt)        # collect loss and metric for training dataset        loss_history["train"].append(train_loss)        metric_history["train"].append(train_metric)Then, we will evaluate the model on the validation dataset:        # evaluate model on validation dataset        model.eval()        with torch.no_grad():
    loss_history["train"].append(train_loss)
    metric_history["train"].append(train_metric)

    model.eval()
    with torch.no_grad():
        val_loss,val_metric=loss_epoch(model,loss_func,val_dl,sanity_check)
        loss_history["val"].append(val_loss)
        metric_history['val'].append(val_metric)

    if val_loss < best_loss:
        best_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), path2weights)
        print("copied best model weights")
    
    lr_scheduler.step(val_loss)
    if current_lr != get_lr(opt):
        print("loading best model weights")
        model.load_state_dict(best_model_wts)

    print("train loss: %.6f, dev loss: %.6f, accuracy: %.2f"%(train_loss,val_loss,100*val_metric))
    print("_"*10)

import copy
loss_func = nn.NLLLoss(reduction="sum")
opt = optim.Adam(cnn_model.parameters(), lr=3e-4)
lr_scheduler = ReduceLROnPlateau(opt, mode='min',factor=0.5,patience=20,verbose=1)

params_train={
    "num_epochs": 100,
    "optimizer": opt, "loss_func": loss_func, "train_dl": train_dl, "val_dl": val_dl, "sanity_check": True, "lr_scheduler": lr_scheduler, "path2weights": "./models/weights.pt",
}

cnn_model,loss_hist,metric_hist=train_val(cnn_model,params_train)