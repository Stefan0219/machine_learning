import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from scipy import stats
import matplotlib.pyplot as plt
def distance(p1:np.ndarray, p2:np.ndarray): 
    assert p1.shape == p2.shape
    po = p1-p2
    return np.sqrt((po*po).sum())
    
def acc(a1:np.ndarray, a2:np.ndarray):
    assert a1.shape == a2.shape
    return np.mean(a1==a2)
def knn(f):
    k = f
    ds = load_digits()
    result = []
    X_train, X_test, y_train, y_test = train_test_split(ds['data'],ds['target'])
    for i in X_test:
        dist = []
        for j in X_train:
            dist.append(distance(i,j))
        temp = []
        d = dist.copy()
        y = y_train.copy()
        for _ in range(k):
            minIndex = np.argmin(d)
            temp.append(y_train[minIndex])
            np.delete(d,minIndex)
            np.delete(y,minIndex)
        b = stats.mode(temp)[0][0]
        s = np.bincount(temp)
        maxIns = np.argmax(s)
        assert b == maxIns
        result.append(b)   
    result = np.array(result)
    return acc(result,y_test)

p = []
for i in range(1,10):
    p.append(knn(2))
plt.plot(p)
plt.title("ACC")
plt.show()
print(np.mean(p))
