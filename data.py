import pandas as pd
import torch
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

data_path=r'data.csv'
elements=['收盘价','最高价','最低价','开盘价','前收盘']
# element=['开盘价']

def single_data():#以收盘价为y，且x归一化

    data_all = pd.read_csv(data_path, encoding='gbk')
    data_all = data_all[data_all['收盘价']!= 0]
    X = data_all[elements].values[::-1].copy()
    y = X[5:,0] # 从初始天后的第11天开始的收盘价 y.shape = (5083,)
    y = y.reshape(-1,1) # y.shape = (5083,1)
    X = X[:-5,:]
    X_normalized = (X - np.mean(X,axis=0)) / np.std(X,axis=0)
    return X_normalized,y


def data_load(seq_len):
    x,y=single_data()
    len=x.shape[0]
    data_last_index=len-seq_len
    X=[]
    Y=[]
    for i in range(0,data_last_index,seq_len):
        data_x=np.expand_dims(x[i:i+seq_len],0)   #[1,seq,feature_size]
        data_y=np.expand_dims(y[i:i+seq_len],0)   #[1,seq,out_size]
        # data_y=np.expand_dims(y[,0)   #[1,seq,out_size]
        X.append(data_x)
        Y.append(data_y)
    data_x= np.concatenate(X, axis=0)
    data_y=np.concatenate(Y, axis=0)
    data=torch.from_numpy(data_x).type(torch.float32)
    label=torch.from_numpy(data_y).type(torch.float32)
    return data,label    #[num_data,seq,feature_size]  [num_data,seq] 默认out_size为1

def dataset(seq_len,batch):
    X,Y=data_load(seq_len)
    feature_size=X.shape[-1]
    out_size=Y.shape[-1]
    train_X, test_X,train_y, test_y = train_test_split(X,Y, test_size=0.2,shuffle=False)
    dataset_train=TensorDataset(train_X,train_y)
    dataset_test=TensorDataset(test_X,test_y)
    dataloader_train=DataLoader(dataset_train,batch_size=batch,shuffle=False)
    dataloader_test=DataLoader(dataset_test,batch_size=batch,shuffle=False)
    return dataloader_train,dataloader_test,feature_size,out_size
