#  Copyright 2018 Alvaro Villoslada (Alvipe)
#  This file is part of Open Myo.
#  Open Myo is distributed under a GPL 3.0 license

from emgesture import fextraction as fex
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pickle

def save_data(data):
    with open("prediction.pkl", 'wb') as fp:
        pickle.dump(data, fp)

# Data loading
with open("../emg_data/emg_data_20190806-161339.pkl",'rb') as fp:
    emg_data = pickle.load(fp)

with open("../emg_data/emg_data_20190806-162404.pkl",'rb') as fp:
    emg_test_data = pickle.load(fp)

# the number of gestures
n_classes = len(emg_data)
# the number of times each gesture is performed
n_iterations = [len(value) for value in emg_data.values()][0]
# the number of data points per gesture - 8 for EMG, 4 for IMU
n_channels = 12
n_signals = n_classes*n_iterations*n_channels
emg = list()
segmented_emg = list()
class_labels = list()

n_iterations_test = [len(value) for value in emg_test_data.values()][0]
n_signals_test = n_classes*n_iterations_test*n_channels
emg_test = list()
segmented_emg_test = list()

#for m in range(1,n_classes+1):
#    for i in range(n_iterations):
#        for c in range(1,n_channels+1):
#            emg.append(emg_data['motion'+str(m)+'_ch'+str(c)][:,i]) #motion1_ch1_i1, motion1_ch2_i1, motion1_ch1_i2, motion1_ch2_i2

# going to do this manually for now just to make sure that the prediction matrices from both modules line up
class_labels.append('man')
class_labels.append('skimmer')
class_labels.append('music')
class_labels.append('green')
class_labels.append('bright')
class_labels.append('drawer')
class_labels.append('away')
class_labels.append('learn')
class_labels.append('accept')

class_index = list()
class_index.append('accept')
class_index.append('away')
class_index.append('bright')
class_index.append('drawer')
class_index.append('green')
class_index.append('learn')
class_index.append('man')
class_index.append('music')
class_index.append('skimmer')

# loop through each gesture
for g in class_labels:
# loop through each iteration of each gesture
    for i in range(n_iterations):
# loop through all 12 data points from each iteration
        for c in range(n_channels):
# create a list of arrays, where, for example, list(zip(*emg_data[0][1]))[2] would be the 2nd iteration of the 1st gesture, and the 3rd (of 8) EMG reading
            emg.append(np.array(list(zip(*emg_data[g][i]))[c][0:999]))

for g in class_labels:
    for i in range(n_iterations_test):
        for c in range(n_channels):
            emg_test.append(np.array(list(zip(*emg_test_data[g][i]))[c][0:999]))

#for z in range(n_signals):
#    emg[z] = emg[z]*(5/2)/2**24

# Segmentation - 72 arrays, 1 for each data point from each individual gesture (12*6)
for n in range(n_signals):
    segmented_emg.append(fex.segmentation(emg[n],n_samples=50))

for n in range(n_signals_test):
    segmented_emg_test.append(fex.segmentation(emg_test[n],n_samples=50))

# Feature calculation
feature_list = [fex.mav, fex.rms, fex.var, fex.ssi, fex.zc, fex.wl, fex.ssc, fex.wamp]

# get minimum length of segments (should be 9)
n_segments = len(segmented_emg[0][0])
for i in range(0,n_signals,n_channels):
    if len(segmented_emg[i][0]) < n_segments :
        n_segments = len(segmented_emg[i][0])

n_segments_test = len(segmented_emg_test[0][0])
for i in range(0,n_signals_test,n_channels):
    if len(segmented_emg_test[i][0]) < n_segments_test :
        n_segments_test = len(segmented_emg_test[i][0])

# get length of feature list (should be 8)
n_features = len(feature_list)
# initialize a 54 x 96 matrix of 0s
feature_matrix = np.zeros((n_classes*n_iterations*n_segments,n_features*n_channels))
feature_matrix_test = np.zeros((n_classes*n_iterations_test*n_segments_test,n_features*n_channels))

n = 0
# loop - 0, 12, 24, 36, 48, 60, stop at 72
for i in range(0,n_signals,n_channels):
# j is 0 through 8 - the index of the column in the array
    for j in range(n_segments):
# n goes from 0 to 54, where each index is a 96 element array -
# so the first 9 will be for gesture 1, then gesture 2, etc.
        feature_matrix[n] = fex.features((segmented_emg[i][:,j],
                                          segmented_emg[i+1][:,j],
                                          segmented_emg[i+2][:,j],
                                          segmented_emg[i+3][:,j],
                                          segmented_emg[i+4][:,j],
                                          segmented_emg[i+5][:,j],
                                          segmented_emg[i+6][:,j],
                                          segmented_emg[i+7][:,j],
                                          segmented_emg[i+8][:,j],
                                          segmented_emg[i+9][:,j],
                                          segmented_emg[i+10][:,j],
                                          segmented_emg[i+11][:,j]),feature_list)
        n = n + 1

k = 0
for i in range(0,n_signals_test,n_channels):
    for j in range(n_segments_test):
        feature_matrix_test[k] = fex.features((segmented_emg_test[i][:,j],
                                               segmented_emg_test[i+1][:,j],
                                               segmented_emg_test[i+2][:,j],
                                               segmented_emg_test[i+3][:,j],
                                               segmented_emg_test[i+4][:,j],
                                               segmented_emg_test[i+5][:,j],
                                               segmented_emg_test[i+6][:,j],
                                               segmented_emg_test[i+7][:,j],
                                               segmented_emg_test[i+8][:,j],
                                               segmented_emg_test[i+9][:,j],
                                               segmented_emg_test[i+10][:,j],
                                               segmented_emg_test[i+11][:,j]),feature_list)
        k = k + 1

# Target matrix generation
y_train = fex.generate_target(n_iterations*n_segments,class_labels)
y_test = fex.generate_target(n_iterations_test*n_segments_test,class_labels)

# Dimensionality reduction and feature scaling - 9 data points for each gesture
[X_train,reductor,scaler] = fex.feature_scaling(feature_matrix, y_train)
[X_test,reductor_test,scaler_test] = fex.feature_scaling(feature_matrix_test, y_test)

# Split dataset into training and testing datasets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
print(X_train)
print(y_train)
print(X_test)
print(y_test)

# Classifier training
classifier = SVC(kernel='rbf',C=10,gamma=10)
classifier.fit(X_train,y_train)

# Classification
predict = classifier.predict(X_test)
predictions = np.zeros((n_classes*n_iterations_test,n_classes))
for i in range(n_classes):
    for j in range(n_iterations_test):
        class_array = np.zeros(n_classes)
        for k in range(n_segments_test):
            for l in range(n_classes):
                if (predict[i*n_segments_test*n_iterations_test+j*n_segments_test+k] == class_index[l]):
                    class_array[l] = class_array[l] + 1
        for l in range(n_classes):
            predictions[i*n_iterations_test+j,l] = class_array[l]/n_segments_test
print(predict)
print(predictions)
#save_data(predictions)
print("Classification accuracy = %0.5f." %(classifier.score(X_test,y_test)))

## Cross validation (optional; takes a lot of time)
#from sklearn.cross_validation import StratifiedShuffleSplit
#from sklearn.grid_search import GridSearchCV
#from sklearn.svm import SVC
#
#C_range = np.logspace(-5,5,11)
#gamma_range = np.logspace(-30,1,32)
#param_grid = dict(gamma=gamma_range,C=C_range)
#cv = StratifiedShuffleSplit(y, n_iter=20,test_size=0.2,random_state=42)
#grid = GridSearchCV(SVC(),param_grid=param_grid,cv=cv)
#grid.fit(X,y)
#print("The best parameters are %s with a score of %0.2f" % (grid.best_params_,grid.best_score_))
#print("%d" % len(class_labels))
plt.scatter(X_train[0:n_segments*n_iterations,0],X_train[0:n_segments*n_iterations,1],c='red',label=class_labels[0])
plt.scatter(X_test[0:n_segments_test*n_iterations_test,0],X_test[0:n_segments_test*n_iterations_test,1],c='red')
plt.scatter(X_train[n_segments*n_iterations:2*n_segments*n_iterations,0],X_train[n_segments*n_iterations:2*n_segments*n_iterations,1],c='blue',label=class_labels[1])
plt.scatter(X_test[n_segments_test*n_iterations_test:2*n_segments_test*n_iterations_test,0],X_test[n_segments_test*n_iterations_test:2*n_segments_test*n_iterations_test,1],c='blue')
plt.scatter(X_train[2*n_segments*n_iterations:3*n_segments*n_iterations,0],X_train[2*n_segments*n_iterations:3*n_segments*n_iterations,1],c='green',label=class_labels[2])
plt.scatter(X_test[2*n_segments_test*n_iterations_test:3*n_segments_test*n_iterations_test,0],X_test[2*n_segments_test*n_iterations_test:3*n_segments_test*n_iterations_test,1],c='green')
plt.scatter(X_train[3*n_segments*n_iterations:4*n_segments*n_iterations,0],X_train[3*n_segments*n_iterations:4*n_segments*n_iterations,1],c='cyan',label=class_labels[3])
plt.scatter(X_test[3*n_segments_test*n_iterations_test:4*n_segments_test*n_iterations_test,0],X_test[3*n_segments_test*n_iterations_test:4*n_segments_test*n_iterations_test,1],c='cyan')
plt.scatter(X_train[4*n_segments*n_iterations:5*n_segments*n_iterations,0],X_train[4*n_segments*n_iterations:5*n_segments*n_iterations,1],c='magenta',label=class_labels[4])
plt.scatter(X_test[4*n_segments_test*n_iterations_test:5*n_segments_test*n_iterations_test,0],X_test[4*n_segments_test*n_iterations_test:5*n_segments_test*n_iterations_test,1],c='magenta')
plt.scatter(X_train[5*n_segments*n_iterations:6*n_segments*n_iterations,0],X_train[5*n_segments*n_iterations:6*n_segments*n_iterations,1],c='lime',label=class_labels[5])
plt.scatter(X_test[5*n_segments_test*n_iterations_test:6*n_segments_test*n_iterations_test,0],X_test[5*n_segments_test*n_iterations_test:6*n_segments_test*n_iterations_test,1],c='lime')
plt.scatter(X_train[6*n_segments*n_iterations:7*n_segments*n_iterations,0],X_train[6*n_segments*n_iterations:7*n_segments*n_iterations,1],c='orange',label=class_labels[6])
plt.scatter(X_test[6*n_segments_test*n_iterations_test:7*n_segments_test*n_iterations_test,0],X_test[6*n_segments_test*n_iterations_test:7*n_segments_test*n_iterations_test,1],c='orange')
plt.scatter(X_train[7*n_segments*n_iterations:8*n_segments*n_iterations,0],X_train[7*n_segments*n_iterations:8*n_segments*n_iterations,1],c='yellow',label=class_labels[7])
plt.scatter(X_test[7*n_segments_test*n_iterations_test:8*n_segments_test*n_iterations_test,0],X_test[7*n_segments_test*n_iterations_test:8*n_segments_test*n_iterations_test,1],c='yellow')
plt.scatter(X_train[8*n_segments*n_iterations:9*n_segments*n_iterations,0],X_train[8*n_segments*n_iterations:9*n_segments*n_iterations,1],c='purple',label=class_labels[8])
plt.scatter(X_test[8*n_segments_test*n_iterations_test:9*n_segments_test*n_iterations_test,0],X_test[8*n_segments_test*n_iterations_test:9*n_segments_test*n_iterations_test,1],c='purple')
#plt.scatter(X[3*n_segments*n_iterations:4*n_segments*n_iterations,0],X[3*n_segments*n_iterations:4*n_segments*n_iterations,1],c='cyan',label=class_labels[3])
#plt.scatter(X[4*n_segments*n_iterations:5*n_segments*n_iterations,0],X[4*n_segments*n_iterations:5*n_segments*n_iterations,1],c='magenta',label=class_labels[4])
#plt.scatter(X[5*n_segments*n_iterations:6*n_segments*n_iterations,0],X[5*n_segments*n_iterations:6*n_segments*n_iterations,1],c='lime',label=class_labels[5])
#plt.scatter(X[6*n_segments*n_iterations:7*n_segments*n_iterations,0],X[6*n_segments*n_iterations:7*n_segments*n_iterations,1],c='orange',label=class_labels[6])
plt.legend(scatterpoints=1,loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
