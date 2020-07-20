import numpy as np

## 1.1
# arr = np.ceil(np.random.rand(20) * 6) 
# print (arr)

## 1.2
y = np.array([11,22,33,44,55,66])
z = y.reshape((round(len(y)/2),2))
#print (z)

## 1.3
x = np.max(z)
#print(x)
r = np.where(z==x)[0][0]
c = np.where(z==x)[1][0]
#print(r)
#print(c)

## 1.4
v = np.array([1,4,7,1,2,6,8,7,9])
x = np.where(v==1, 1, 0).sum()
print(x)
