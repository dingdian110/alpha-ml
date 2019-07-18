import numpy as np

a=np.array([1,2,3,4,5])
print(a.shape)
b=np.zeros([5,1])
b[:,:]=a
print(b)
a=np.ones((2,5))
print(a)
b[0:2,:]=a
print(b)