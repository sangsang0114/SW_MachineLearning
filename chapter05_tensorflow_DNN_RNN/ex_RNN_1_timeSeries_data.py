#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

def make_data() :
    """
    Keras 에서 RNN 을 사용하려면 입력 데이터는 
    (nb_samples, timesteps, input_dim) 크기를 가지는 
    ndim=3인 3차원 텐서(tensor) 형태이어야 한다.

    nb_samples: 자료의 수
    timesteps: 순서열의 길이
    input_dim: x 벡터의 크기
    여기에서는 단일 시계열이므로 input_dim = 1 이고 
    3 스텝 크기의 순서열을 사용하므로 timesteps = 3 이며 자료의 수는 18 개이다.

    다음코드와 같이 원래의 시계열 벡터를 Toeplitz 행렬 형태로 변환하여 3차원 텐서를 만든다.
    """
    
    DataDf = pd.read_csv("data.csv") 
    Data = DataDf.to_numpy() 
    
     
    X_train = Data[:-1, :3][:, :, np.newaxis]  
                # np.newaxis : numpy array의 차원을 늘려줌. 1D->2D,2D->3D,3D->4D 
    Y_train = Data[:-1, 3] 
    
    
    print("X_train.shape, Y_train.shape  = ", X_train.shape, Y_train.shape  ) 
    
    return X_train, Y_train


# In[17]:


def show_graph(X_train, Y_train) :
    
    plt.subplot(211)
    plt.plot([0, 1, 2], X_train[0].flatten(), 'bo-', label="input sequence")
    plt.plot([3], Y_train[0], 'ro', label="target")
 
    plt.legend()
    plt.title("First sample sequence")


if __name__ == '__main__':
    X_train, Y_train =  make_data()
    show_graph( X_train, Y_train  ) 


# In[ ]:




