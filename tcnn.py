<<<<<<< HEAD
# Python code to read in time ordered expectation values of QMB system observables in the short time 
# domain and extrapolate values out to the long time domain using CNN   
#
# Justin Reyes : 10/3/2022
#
# Collaborators: Sayandip Dhara, Eduardo Mucciolo 


#import libraries 

=======
>>>>>>> origin/qmb_tcnn
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy import random
import tensorflow as tf
import numpy as np
import time
import os
import argparse
import matplotlib.pyplot as plt
import os.path

<<<<<<< HEAD

#parse command line arguments
=======
>>>>>>> origin/qmb_tcnn
parser=argparse.ArgumentParser(description="MLP Regression of operator expectation values")
parser.add_argument("--data_dir",help="Path to directory of formatted data files")
parser.add_argument("--nepochs",type=int,help="number of epochs for training")
parser.add_argument("--train_size",type=int,help="number of data points to use for training")
parser.add_argument("--saved_model",help="y if there is a saved model, n if there is not")
args=parser.parse_args()

<<<<<<< HEAD

#plot the absolute error between the extrapolated data and the true value 
=======
>>>>>>> origin/qmb_tcnn
def plot_eps(name,dt,offset,tau):
    f='mlp_sz_vs_mps_sz2.d'
    ifp=open(f,'r')
    d1=np.loadtxt(ifp)
    eps=[]
    for i in range(len(d1)):
        eps.append(abs(d1[i][2]-d1[i][1]))
    ofp=open(name+'_eps.d','w')
    for i in range(len(d1)):
        ofp.write(str(dt*i + offset)+'\t'+str(eps[i])+'\n')
    ofp.close()

    fig,ax=plt.subplots()
    ax.plot(dt*d1[:,0]+offset,eps,'x',lw=3,label=r'$\epsilon$')
    #plt.legend(loc='best',fontsize=24)
    plt.yscale('log')
    plt.ylim([1e-5,1e-1])
    plt.xlim([0,tau])
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.xlabel(r'$\tau$',fontsize=28)
    plt.ylabel(r'$\epsilon$',fontsize=28)
    ax.grid(which='major',axis='both',linestyle='--')
    plt.tight_layout()
    plt.show()

    
<<<<<<< HEAD
#plot the extrapolated data and the true value 
=======
>>>>>>> origin/qmb_tcnn
def plot_data(window_len,dt):
    f1='sz_mps_vs_sz_cnn.d'

    ifp=open(f1,'r')
    d1=np.loadtxt(ifp)
    x=10*len(d1)*[dt*window_len]
    y=[]
    for i in range(5*len(d1)):
        y.append(random.random())
        y.append(-1*random.random())
    fig,ax=plt.subplots()
    ax.plot(d1[:,0],d1[:,1],'x',lw=3,label=r'CNN')
    ax.plot(d1[:,0],d1[:,2],'o',markerfacecolor='none',lw=3,label=r'MPS')
    ax.plot(x,y,'.',lw=1,label=r'Input boundary')
#    ax.set_yticks(ax.get_yticks()[::1])    
    plt.legend(loc='best', fontsize=14)
    plt.ylim([-1,1])
    plt.xlim([0,20])
    plt.xlabel(r'$\frac{\tau}{J}$',fontsize=28)
    plt.ylabel(r'$\langle S^z \rangle$', fontsize=28)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    ax.grid(which='major',axis='both',linestyle='--')
    plt.tight_layout()
    plt.show()
 
<<<<<<< HEAD

#create NN model and train over the number of epochs with the given batch size   
=======
   
>>>>>>> origin/qmb_tcnn
def pCNN3(xtrain0,xtrain1,ytrain,fit,epochs,batch_size):
#    print("training with {} input datasets with fitting windows of size {}".format(len(xtrain0),len(xtrain0[0])))
    n0=len(xtrain0[0])
    n1=len(xtrain1[0])
    m=len(ytrain[0])
    print("training with {} input datasets with fitting windows of size {} predicting {} values".format(len(xtrain0),n0,m))

<<<<<<< HEAD
#define channel 1 of the CNN 
=======
>>>>>>> origin/qmb_tcnn
    input0=tf.keras.Input(shape=(n0,1))
    conv0a=tf.keras.layers.Conv1D(filters=100,kernel_size=10,strides=5,activation='relu')(input0)
#    conv0b=tf.keras.layers.Conv1D(filters=20,kernel_size=2padding='same')(conv0a)
    pool0=tf.keras.layers.AveragePooling1D(pool_size=2)(conv0a)
    flat0=tf.keras.layers.Flatten()(pool0)

<<<<<<< HEAD
#define channel 2 of the CNN 
=======
>>>>>>> origin/qmb_tcnn
    input1=tf.keras.Input(shape=(n1,1))
    conv1a=tf.keras.layers.Conv1D(filters=100,kernel_size=10,strides=5,activation='relu')(input1)
#    conv1b=tf.keras.layers.Conv1D(filters=20,kernel_size=2,padding='same')(conv1a)
    pool1=tf.keras.layers.AveragePooling1D(pool_size=2)(conv1a)
    flat1=tf.keras.layers.Flatten()(pool1)
    
<<<<<<< HEAD

#merge the channels to pass into the fully-connected layer 
    merged=tf.keras.layers.concatenate([flat0,flat1])
    
#define the fully connected layer 
=======
    merged=tf.keras.layers.concatenate([flat0,flat1])
    
>>>>>>> origin/qmb_tcnn
#    dense1=tf.keras.layers.Dense(units=n0+n1,activation='linear')(merged)
#    drop1=tf.keras.layers.Dropout(0.25)(dense1)
#    dense2=tf.keras.layers.Dense(units=2*(n0+n1), activation='relu')(drop1)
#    drop2=tf.keras.layers.Dropout(0.25)(dense2)
    output=tf.keras.layers.Dense(units=m, activation='linear')(flat0)


<<<<<<< HEAD
#compile the model 
=======
>>>>>>> origin/qmb_tcnn
    model=tf.keras.Model(inputs=[input0,input1],outputs=output)
    model.compile(loss='mse',optimizer='adam',metrics=['mae'])
    model.summary()

<<<<<<< HEAD
#train the model 
=======
>>>>>>> origin/qmb_tcnn
    model.fit([xtrain0[:fit],xtrain1[:fit]],ytrain[:fit],epochs=epochs,batch_size=batch_size)
#    ls,ms=0,0
#    print(len(xtrain0[fit:]),len(xtrain1[fit:]),len(ytrain[fit:]))
    ls,ms=model.evaluate([xtrain0[fit:],xtrain1[fit:]],ytrain[fit:],batch_size=batch_size)
    return model,ls,ms

def read_data(filename,train_size):
    print("reading file: {}".format(filename))
    px1=[]
    px2=[]
    py=[]
<<<<<<< HEAD

#uncomment the correct directory name to read the data from 
=======
>>>>>>> origin/qmb_tcnn
    d=np.loadtxt('12_tfim_hx_0.2_0.8/'+filename)
#    d=np.loadtxt('10_tfim_hx_0.2_0.8/'+filename)
#    d=np.loadtxt('tfim_data/'+filename)
#    d=np.loadtxt('tfim_data_08_11_21/12_tfim_data/'+filename)
#    d=np.loadtxt('tfim_data_08_11_21/14_tfim_data/'+filename)
#    d=np.loadtxt('data_XXZ_W1_100/data_12_w1_rand_new/'+filename)
#    d=np.loadtxt('data_XXZ_W1_100/data_14_w1_rand_new/'+filename)
#    d=np.loadtxt('data_XXZ_J2Del1/'+filename)
    p=[]
    ii=0
    for k in d:
        #for xxz file read column 1
        #for tfim file read colum 2
        if (ii < train_size):
            px1.append(k[2])
            px2.append(k[1])
        else:
            py.append(k[2])
        
        ii+=1  

    
    return px1,px2,py 
<<<<<<< HEAD


#get the datasets from the main directory  
=======
 
>>>>>>> origin/qmb_tcnn
def get_datasets(directory,train_size):
    train_x1=[]
    train_x2=[]
    train_y=[]
    for i in os.listdir(directory):
        x1,x2,y=read_data(i,train_size) 
        train_x1.append(x1)
        train_x2.append(x2)
        train_y.append(y)

    train_x1=np.array(train_x1)
    train_x2=np.array(train_x2)
    train_y=np.array(train_y)

    print(len(train_x1[0]),len(train_y[0]))
    return train_x1,train_x2,train_y



# file for Ising is 10_tfim_t_50.out --> read first 50 values column 2
# file for XXZ is 12_XXZ_j2_del1_mps.out --> read first 100 values column 1
# file for Ising is 20_tfim_Crit_Del4_hx2.out --> read first ? values column 1

<<<<<<< HEAD

#train the model directly or load an already trained model saved in the directory as 'tcnn.h5' 
=======
>>>>>>> origin/qmb_tcnn
def train_model(data_dir,train_size,nepochs):
    train_x1,train_x2,train_y=get_datasets(data_dir,train_size)
    hx=[]
    
    if (args.saved_model == 'n'):
        t1=time.time()
        model,ls,ms=pCNN3(train_x1,train_x2,train_y,int(0.8*len(train_x1)),nepochs,1)
        t2=time.time()
        print("time to train model {}".format(t2-t1))    
        model.save("tcnn.h5")
        return model,train_x1,train_x2,train_y
    else:
        model=tf.keras.models.load_model("tcnn.h5")
        return model,train_x1,train_x2,train_y

<<<<<<< HEAD
#test the model and write the test results to a data file  
=======
>>>>>>> origin/qmb_tcnn
def write_data(test_x1,test_x2,test_y,model):
    pred_sz=[]
    t=[]
    err=[]
    dt=0.05
    dh=0.004
    i=random.randint(0,19)
    print("{} --> {} : {} data points".format(i,0.2+dh*(80+i),len(test_x1[i])))
    pred_sz=model.predict([np.array([test_x1[i]]),np.array([test_x2[i]])],batch_size=1)
    rr=0.
    for j in range(len(test_x1[i])):
        t.append(dt*j)
        err.append(0)
        
    for j in range(len(test_y[i])):
        t.append(dt*len(test_x1[i])+dt*(j-1))
        err.append(abs(test_y[i][j] - pred_sz[0][j]))
        rr+=err[-1]
        print("real value: {} vs predicted value: {}".format(test_y[i][j],pred_sz[0][j]))
    rr/=len(t)

    print("MAE for test cases: {}".format(rr))

    ofp1=open('sz_mps_vs_sz_cnn.d','w')
    ofp2=open('sz_cnn_err.d','w')

    for j in range(len(t)):
        if (j < len(test_x1[i])):
            ofp1.write(str(t[j])+'\t'+str(test_x1[i][j])+'\t'+str(test_x1[i][j])+'\n')
        else:
            ofp1.write(str(t[j])+'\t'+str(pred_sz[0][j-len(test_x1[i])])+'\t'+str(test_y[i][j-len(test_x1[i])])+'\n')
        ofp2.write(str(t[j])+'\t'+str(err[j])+'\n')

    ofp1.close()
    ofp2.close()
    return 

<<<<<<< HEAD
#MAIN function calls: train the model, write the data for the model, plot the data  

=======
>>>>>>> origin/qmb_tcnn
model,train_x1,train_x2,train_y=train_model(args.data_dir,args.train_size,args.nepochs)
write_data(train_x1,train_x2,train_y,model)
plot_data(args.train_size,dt=0.05)
