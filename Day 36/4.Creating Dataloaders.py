from torch.utils.data import DataLoader
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_ds = DataLoader(val_ds, batch_size64,shuffle=False)

for x, y in train_dl:
    print(x.shape)
    print(y.shape)
    break

for x, y in val_dl:
    print(x.shape)
    print(y.shape)
    break

#get labels for validation dataset
y_val = [y for _, y in val_ds]

def accuracy(labels, out):
    return np.sum(out==labels)/float(len(labels))

#accuracy all zero predictions
acc_all_zero = accuracy(y_val, np.zeros_like(y_val))
print("accuracy all zero prediction: %2f" %acc_all_zero)

acc_all_ones=accuracy(y_val,np.ones_like(y_val))
print("accuracy all one prediction: %.2f" %acc_all_ones)

acc_random=accuracy(y_val,np.random.randint(2,size=len(y_val)))
print("accuracy random prediction: %.2f" %acc_random)