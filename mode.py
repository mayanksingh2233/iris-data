import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
df = pd.read_csv('iris.data')

x = np.array(df.iloc[:,0:4])
y = np.array(df.iloc[:,4:])

le = LabelEncoder()
y = le.fit_transform(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)

svc = SVC(kernel='linear')
svc.fit(x_train,y_train)

pickle.dump(svc,open('model.pkl','wb'))

