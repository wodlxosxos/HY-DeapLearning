import numpy as np
import random
import time

def sigmoid(z):
    return (1 / (1 + np.exp(-z)))

def loss_func(a,y):
    if y == 1:
        return -np.log10(a + np.exp(-10))
    else:
        return -np.log10(1-a + np.exp(-10))

def linear_func_np(w,b,x):
    return (x@w + b)


# w의 초기값으로 0도 넣어보기.
#초기 설정.
m = 10000
n = 500
K = 5000
alpha = 6.01
dataset = np.load('./dataset.npz')

x1_train = dataset['x1_train']
x2_train = dataset['x2_train']
x_train = np.zeros((m,2))
x_train[:,0] = x1_train
x_train[:,1] = x2_train
y_train = [0]*m

x1_test = dataset['x1_test']
x2_test = dataset['x2_test']
x_test = np.zeros((n,2))
x_test[:,0] = x1_test
x_test[:,1] = x2_test
y_test = [0]*n

# y값 설정.
for i in range(m):
    if (x1_train[i] + x2_train[i]) > 0:
        y_train[i] = 1

for j in range(n):
    if (x1_test[j] + x2_test[j]) > 0:
        y_test[j] = 1
        
# w값 설정
w = np.array([10.0,7.0])
b = 5

# 알고리즘 구현
start= time.time()
train_cost = 0
for try_num in range(K):
    z_list = linear_func_np(w,b,x_train)
    z_list = sigmoid(z_list)
    
    J = np.sum(loss_func(z_list, y_train))
    dz = z_list - y_train
    dw = x_train.T@dz
    db = np.sum(dz)
    
    J /= m
    dw /= m
    db /= m
    w -= alpha*dw
    b -= alpha*db
    train_cost += J
end_train = time.time()

#-----------------test-------
succ = 0
fail = 0
z_list = linear_func_np(w,b,x_train)
z_list = sigmoid(z_list)
for try_num in range(m):
    a = z_list[try_num]
    if a > 0.5 :
        result = 1
    else:
        result = 0
    if result == y_train[try_num]:
        succ += 1
    else:
        fail += 1
print("Accuracy(train) : ", (succ/m)*100,"%")

succ = 0
fail = 0
z_list = linear_func_np(w,b,x_test)
z_list = sigmoid(z_list)
for try_num in range(n):
    a = z_list[try_num]
    if a > 0.5 :
        result = 1
    else:
        result = 0
    if result == y_test[try_num]:
        succ += 1
    else:
        fail += 1
print("Accuracy(test) : ", (succ/n)*100, "%")
end_test = time.time()
print("train time : ", end_train - start, ", test time : ", end_test-end_train)
