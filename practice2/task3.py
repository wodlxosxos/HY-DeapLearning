import numpy as np
import time

def sigmoid(z):
    return (1 / (1 + np.exp(-z)))

def loss_func(a,y):
    if y == 1:
        return -np.log10(a + np.exp(-10))
    else:
        return -np.log10(1-a + np.exp(-10))

def diff_sigmoid(z):
    return (sigmoid(z)*(1-sigmoid(z) + np.exp(-10)))

def linear_func_np(w,b,x):
    return (x@w.T + b)


# w의 초기값으로 0도 넣어보기.
#초기 설정.
m = 10000
n = 500
K = 5000
alpha = 0.03
dataset = np.load('./dataset.npz')

x_train = np.zeros((m,2))
x_train[:,0] = dataset['x1_train']
x_train[:,1] = dataset['x2_train']
y_train = [0]*m

x_test = np.zeros((n,2))
x_test[:,0] = dataset['x1_test']
x_test[:,1] = dataset['x2_test']
y_test = [0]*n

# y값 설정.
for i in range(m):
    if np.sum(x_train[i]) > 0:
        y_train[i] = 1

for j in range(n):
    if np.sum(x_test[j]) > 0:
        y_test[j] = 1
        
# w값 설정
w1 = np.array([[0.9444385, 4.40207366], [8.78037246, 2.73316276],
               [9.57849742, 2.74094659]])
w2 = np.array([[4.6221737, 9.55349149, 9.53626129]])
b1 = np.array([[4.82043275], [0.42613082], [8.20532193]])
b2 = np.array([[1.2345103]])

# 알고리즘 구현 --------------------------------
start = time.time()
for try_num in range(K):
    z1_list = linear_func_np(w1,b1.T,x_train)
    a1_list = sigmoid(z1_list)
    
    z2_list = linear_func_np(w2,b2,a1_list)
    a2_list = sigmoid(z2_list)
    
    J = np.sum(loss_func(a2_list, y_train))
    
    dz2 = a2_list.T - y_train
    dw2 = dz2@a1_list
    db2 = np.sum(dz2, axis = 1, keepdims = True)
    dw2 /= m
    db2 /= m

    dz1 = (w2.T@dz2).T*diff_sigmoid(z1_list)
    dw1 = x_train.T@dz1
    db1 = np.sum(dz1, keepdims = True)

    w2 -= alpha*dw2
    b2 -= alpha*db2
    
    J /= m
    dw1 /= m
    db1 /= m
    w1 -= alpha*dw1.T
    b1 -= alpha*db1.T
    if try_num%1000 == 99999:
        print(J)

end_train = time.time()

# test ------------------------------------------
succ = 0
fail = 0
z_list = linear_func_np(w1,b1.T,x_train)
z_list = sigmoid(z_list)
z_list = linear_func_np(w2,b2,z_list)
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
z_list = linear_func_np(w1,b1.T,x_test)
z_list = sigmoid(z_list)
z_list = linear_func_np(w2,b2,z_list)
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
