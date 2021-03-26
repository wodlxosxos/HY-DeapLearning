import numpy as np
import random
import time

start = time.time()
def linear_func(w,b,x):
    return (w[0]*x[0] + w[1]*x[1] + b)

def sigmoid(z):
    return (1 / (1 + np.exp(-z)))

def loss_func(a,y):
    if y == 1:
        return -np.log10(a)
    else:
        return -np.log10(1-a)

def linear_func_np(w,b,x):
    return (w@x + b)


# w의 초기값으로 0도 넣어보기.
#초기 설정.
m = 1000
n = 100
alpha = 6.01
dataset = np.load('./dataset.npz')

x1_train = dataset['x1_train']
x2_train = dataset['x2_train']
x_train = np.zeros((1000,2))
x_train[:,0] = x1_train
x_train[:,1] = x2_train
y_train = [0]*m

x1_test = dataset['x1_test']
x2_test = dataset['x2_test']
x_test = np.zeros((100,2))
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
for try_num in range(2000):
    J = 0
    dw1 = 0
    dw2 = 0
    db = 0
    for i in range(m):
        z = linear_func(w, b, x_train[i])
        a = sigmoid(z)# a = pred_y , y_hat
        J += loss_func(a, y_train[i])
        dz = a - y_train[i]
        dw1 += x_train[i][0]*dz
        dw2 += x_train[i][1]*dz
        db += dz
    J /= m
    dw1 /= m
    dw2 /= m


    db /= m
    w[0] -= alpha*dw1
    w[1] -= alpha*dw2
    b -= alpha*db

result = -1
succ = 0
fail = 0
for try_num in range(100):
    z = linear_func(w,b,x_test[try_num])
    a = sigmoid(z)
    if a > 0.5 :
        result = 1
    else:
        result = 0
    if result == y_test[try_num]:
        succ += 1
    else:
        fail += 1
print("Accuracy : ", succ, "%")
print("실행시간(element-wise version) : ", time.time()-start)
