import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors
#from numpy import genDataSet
from sklearn.model_selection import KFold
import time

abc = []
a = []
b = []
c = []
# read digits data & split it into X (training input) and y (target output)
for i in range(0,100):
    def genDataSet (N) :
        X = np.random.normal(0, 1, N)
        ytrue = (np.cos(X) + 2) / (np.cos(X * 1.4) + 2)
        noise = np.random.normal(0, 0.2, N)
        y = ytrue + noise
        return X, y, ytrue
    X ,y, ytrue = genDataSet(1000)
    #plt.plot(X,y, '.')
    #plt.plot(X , ytrue , 'rx' )
    #plt.show()



    X = X.reshape(len(X), 1) #reshaping Training dataset
    bestk = []
    kc = 0

    for n_neighbors in range(1,900,2): #90% Training Data
        kf = KFold(n_splits=10)
        #n_neighbors = 85
        kscore=[]
        k=0

        for train, test in kf.split(X):
            #print("%s %s" % (train, test))
            X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

            #time.sleep(100)

            # we create an instance of Neighbors Classifier and fit the data.
            clf = neighbors.KNeighborsRegressor(n_neighbors, weights='distance')
            clf.fit(X_train, y_train)

            kscore.append(clf.score(X_test,y_test))
            #print kscore[k]
            k=k+1

            #print (n_neighbors)
        bestk.append(sum(kscore)/len(kscore))
        #print bestk[kc]
        kc+=1


    # to do here: given this array of E_outs in CV, find the max, its
    # corresponding index, and its corresponding value of n_neighbors
    #print "Eout="
    #print clf.score(X,y)
    #print "Eout True = "
    #print clf.score(X,ytrue)

    N_Bestk = sorted(bestk, reverse=True)
    #print N_Bestk
    #print N_Bestk[0], N_Bestk[1], N_Bestk[2]

    index = sorted(range(len(bestk)), key=bestk.__getitem__)
    #print (index[-1]*2)+1   #odd numbers
    #print (index[-2]*2)+1   #odd numbers
    #print (index[-3]*2)+1   #odd numbers

    abc = np.append(abc, [[(index[-1]*2)+1, (index[-2]*2)+1, (index[-3]*2)+1]])

print abc

for i in range (0,300):
    if(i%3 == 0):
        a = np.append(a, [abc[i]])
    elif(i%3 == 1):
        b = np.append(b, [abc[i]])
    else:
        c = np.append(c, [abc[i]])
#print a
#print b
#print c
fig = plt.figure()

#pyplot.hist(a, bins, alpha=0.25)
#pyplot.hist(b, bins, alpha=0.5)
#pyplot.hist(c, bins, alpha=0.75)
#pyplot.show()
#xaxis = range(1, 100)
n, bins, patches = plt.hist(abc, facecolor='red', alpha=0.5)
#n, bins, patches = plt.hist(b, facecolor='blue', alpha=0.5)
#n, bins, patches = plt.hist(c, facecolor='green', alpha=0.5)
#plot_url = py.plot_mpl(fig, filename='docs/mpl-histogram')
plt.title('Result')
plt.xlabel('Num')
plt.ylabel('Hapenning')
plt.axis([0, 900, 0, 150])
plt.show()
