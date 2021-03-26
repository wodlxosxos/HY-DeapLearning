import numpy as np

m = 1000
n = 100

x1_train = np.random.uniform(-10,10,(1,m))[0]
x2_train = np.random.uniform(-10,10,(1,m))[0]

x1_test = np.random.uniform(-10,10,(1,n))[0]
x2_test = np.random.uniform(-10,10,(1,n))[0]

np.savez('./dataset.npz', x1_train=x1_train, x2_train=x2_train,
         x1_test=x1_test,x2_test=x2_test)
