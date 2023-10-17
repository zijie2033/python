import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import glob,cv2
import os
import timeit
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def show_images_labels_predictions(images,labels,predictions,start_id,num=10,latency=0):
    plt.gcf().set_size_inches(15, 35)
    if num>25: num=25
    for i in range(0, num):
        ax=plt.subplot(3,4, i+1)
        ax.imshow(images[start_id], cmap='binary') 
        if( len(predictions) > 0 ) :
            title = 'model_name : ' + str(model_name)
            title += ' model ai = ' + str(predictions[start_id])        
            title += (' (o)' if predictions[start_id]==labels[start_id] else ' (x)')
            title += '\nlabel = ' + str(labels[start_id])
            title +=' latency = ' +str(latency)+' ms'
        else : 
            title = 'label = ' + str(labels[start_id])
        ax.set_title(title,fontsize=12)  
        ax.set_xticks([]);ax.set_yticks([])        
        start_id+=1
    plt.show()
# files = glob.glob("imagedata\*.jpg")  
files=glob.glob('imagedata\\5.jpg')
test_feature=[]
test_label=[]
for file in files:
    img=cv2.imread(file)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
    _, img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV) 
    test_feature.append(img)
    label=file[10:11]  
    test_label.append(int(label))
test_feature=np.array(test_feature)
test_label=np.array(test_label)
test_feature_vector = test_feature.reshape(len(test_feature),28,28,1).astype('float32')
test_feature_normalize = test_feature_vector/255
model_name='base_model.h5'
model = load_model(model_name)
t1 = timeit.default_timer()
for x in range(100):
    prediction=model.predict(test_feature_normalize)
t2 = timeit.default_timer()
t = round(1000 * (t2 - t1), 2)/100

prediction=np.argmax(prediction,axis=1)
show_images_labels_predictions(test_feature,test_label,prediction,0,len(test_feature),t)
