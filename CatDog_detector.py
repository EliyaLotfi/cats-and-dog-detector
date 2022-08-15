import cv2
from sklearn.model_selection import train_test_split
import glob
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump,load
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from joblib import dump

def load_data():
    data_list= []
    labels=[]

    for i,address in enumerate(glob.glob("image\\*\\*")): 
        try:
            img = cv2.imread(address)
            rgb_img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            m,n,c= rgb_img.shape
            img_r=np.reshape(rgb_img,(m,n*c))
            pca=PCA(n_components=10).fit(img_r)
            img_r=pca.transform(img_r)       
            img_r=img_r.flatten()
            x=img_r.size
            c=x//10
            img_r=np.reshape(img_r,(10,c))
            pca=PCA(n_components=10).fit(img_r)
            img_r=pca.transform(img_r)
            x= img_r.max()
            img_r=img_r/x
            img_r=img_r.flatten()
            
            data_list.append(img_r)

            label = address.split("\\")[1]
            labels.append(label)
            if i%100 == 0:
                print("statue: {}/1900 processed".format(i))
        except:
            print("Error")
            continue
    data_list=np.array(data_list)
    x_train,x_test,y_train,y_test=train_test_split(data_list,labels,test_size=0.2,random_state=0)
    return x_train,x_test,y_train,y_test

x_train,x_test,y_train,y_test = load_data()

model = RandomForestClassifier()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)
print("accuracy:{:.2f}".format(accuracy*100))