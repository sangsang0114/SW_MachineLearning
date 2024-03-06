#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

def make_data_Sequence2Sequence( showGraph=False ) :
    """
    Keras 에서 RNN 을 사용하려면 입력 데이터는 
    (nb_samples, timesteps, input_dim) 크기를 가지는 
    ndim=3인 3차원 텐서(tensor) 형태이어야 한다.

    nb_samples: 자료의 수
    timesteps: 순서열의 길이
    input_dim: x 벡터의 크기
    
    여기에서는 단일 시계열이므로 input_dim = 1 이고 
    3 스텝 크기의 순서열을 사용하므로 timesteps = 3 이며 자료의 수는 18 개이다.
 
    """

    DataDf = pd.read_csv("data.csv") 
    Data = DataDf.to_numpy() 

    """
    이번에는 출력값도 3개짜리 순서열로 한다.
    """

    X_train2 = Data[:-3, 0:3][:, :, np.newaxis]
    Y_train2 = Data[:-3, 3:6][:, :, np.newaxis]   ############ 이번에는 출력값도 3개짜리 순서열로 한다.
    print("X_train2.shape, Y_train2.shape  = ", X_train2.shape, Y_train2.shape)
    print("X_train2[0], Y_train2[0]  = ", X_train2[0], Y_train2[0])


    if showGraph:
        plt.subplot(211)
        plt.plot([0, 1, 2], X_train2[0].flatten(), 'bo-', label="input sequence")
        plt.plot([3, 4, 5], Y_train2[0].flatten(), 'ro-', label="target sequence")
        plt.xlim(-0.5, 6.5)
        plt.ylim(-1.1, 1.1)
        plt.legend()
        plt.title("First sample sequence")
  

    return (X_train2,Y_train2)

if __name__ == '__main__':
    make_data_Sequence2Sequence(True)

