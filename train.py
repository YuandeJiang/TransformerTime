from transformerhah import Transformer
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import copy
from data import dataset


NUM_EPOCHS = 200
BS = 16
seq_len = 5
LR = 0.001
png_save_path=r'.\graphs'
if not os.path.isdir(png_save_path):
    os.mkdir(png_save_path)

path_train=os.path.join(png_save_path,'weight.pth')

dataloader_train, dataloader_test, feature_size,out_size = dataset(seq_len,BS)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Transformer().to(device)
criterion = nn.MSELoss()     #忽略 占位符 索引为0.
# optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.99)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_loss=100000
best_epoch=0
# Train loop
for epoch in range(NUM_EPOCHS):
    epoch_loss=0.0

    y_pre=[]
    y_true=[]
    for X,y in dataloader_train:  # enc_inputs : [batch_size, src_len,1](64*5)
        # X: [batch_size,5]
        enc_inputs=X  #(1*64*5)
        # enc_inputs: [1, batch_size, 5]
        # enc_inputs=enc_inputs.squeeze(2)
        # dec_inputs : [batch_size, ]
        # dec_outputs: [batch_size, 1]
        outputs, enc_self_attns = model(enc_inputs)
        # print(outputs.shape)
        # outputs: [batch_size,1]
        # outputs=outputs.squeeze(1)
        # outputs=outputs.unsqueeze(0)
        # y=y.unsqueeze(0)
        # print(y.shape)
        # print(outputs.shape)
        # outputs: [batch_size * tgt_len, tgt_vocab_size]
        loss = criterion(outputs.view(1,-1), y.view(1,-1))
        loss_num=loss.item()
        epoch_loss+=loss_num
        # epoch_loss_avg = epoch_loss / X.size(0)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        y_pre.append(outputs.view(1,-1).detach().numpy())
        y_true.append(y.view(1,-1).detach().numpy())

    if epoch_loss<best_loss:
        best_loss=epoch_loss
        best_epoch=epoch+1
        best_model_wts=copy.deepcopy(model.state_dict())
        torch.save(best_model_wts,path_train)

    pre = np.concatenate(y_pre, axis=1).squeeze(0)  # no norm label
    # print(pre.shape)
    true = np.concatenate(y_true, axis=1).squeeze(0)  # no norm label
    # true=true.squeeze(0)
    if True:
        plt.plot(true, color="blue", alpha=0.5, label = 'true')
        plt.plot(pre, color="red", alpha=0.5, label = 'pre')
        plt.plot(pre - true, color="green", alpha=0.5, label = 'pre - true')
        plt.grid(True, which='both')
        plt.axhline(y=0, color='k')
        plt.legend(loc = 'upper right')
        # pyplot.savefig(os.path.join(png_save_path, 'pre.png'))
        plt.savefig(os.path.join(png_save_path, '%d.png'%epoch))
        plt.close()
    print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(epoch_loss))
print('best_loss::|',best_loss,'---best_epoch::|',best_epoch)
train_over_path=os.path.join(png_save_path,'loss(%d)---epoch(%d).pth'%(best_loss,best_epoch))
os.rename(path_train,train_over_path)

# Evaluate model
model.eval()
eval_loss = 0.0
y_pre=[]
y_true=[]
with torch.no_grad():
    for X,y in dataloader_test:
        enc_inputs=X 
        outputs, enc_self_attns = model(enc_inputs)
        loss = criterion(outputs.view(1,-1), y.view(1,-1))
        eval_loss += loss.item()
        
        y_pre.append(outputs.view(1,-1).detach().numpy())
        y_true.append(y.view(1,-1).detach().numpy())

pre = np.concatenate(y_pre, axis=1).squeeze(0)  # no norm label
true = np.concatenate(y_true, axis=1).squeeze(0)  # no norm label
# true=true.squeeze(0)
if True:
    plt.plot(true, color="blue", alpha=0.5, label = 'true')
    plt.plot(pre, color="red", alpha=0.5, label = 'pre')
    plt.plot(pre - true, color="green", alpha=0.5, label = 'pre - true')
    plt.grid(True, which='both')
    plt.axhline(y=0, color='k')
    plt.legend(loc = 'upper right')
    # pyplot.savefig(os.path.join(png_save_path, 'pre.png'))
    plt.savefig(os.path.join(png_save_path, 'test.png'))
    plt.close()

avg_eval_loss = eval_loss / len(dataloader_test)
# avg_infer_loss = infer_loss / len(test_loader)

print(f"Eval / Infer Loss on test set: {avg_eval_loss:.4f} ")



print('*******************over****************************')