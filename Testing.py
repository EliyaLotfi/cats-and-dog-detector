
from sklearn.decomposition import PCA
import cv2
import numpy as np
import glob
from sklearn.preprocessing import MinMaxScaler
from joblib import load
from sklearn.ensemble import RandomForestClassifier

data=load("cats_dogs2.z")

for address in glob.glob(r""):
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
    pred=data.predict(np.array(img_r).reshape(1,-1))[0]
    cv2.putText(img,pred,(40,40),cv2.FONT_HERSHEY_SIMPLEX,1.9,(255,0,0),10)
    cv2.imshow("image",img)
    cv2.waitKey(0)
cv2.destroyAllWindows()