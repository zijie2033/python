import numpy as np
import os
import matplotlib.pyplot as plt
from keras.models import load_model
import dataset
import timeit
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def show_images_labels_predictions(images,labels,predictions,start_id,num=10,latency=0):
    plt.gcf().set_size_inches(50, 100)
    if num>25: num=25
    for i in range(0, num):
        ax=plt.subplot(7,3, i+1)
        ax.imshow(images[start_id], cmap='binary') 
        if( len(predictions) > 0 ) :
            #title = 'model_name : ' + str(model_name)
            title = ' model ai = ' + str(predictions[start_id])        
            title += (' (o)' if predictions[start_id]==labels[start_id] else ' (x)')
            title += '\nlabel = ' + str(labels[start_id])
            title +=' latency = ' +str(latency)+' ms'
        else : 
            title = 'label = ' + str(labels[start_id])
        ax.set_title(title,fontsize=10)  
        ax.set_xticks([]);ax.set_yticks([])        
        start_id+=1
    plt.show()
test_images, test_labels = dataset.load_data('inference_data',False)
test_images, test_labels = dataset.shuffle_data(test_images, test_labels)
test_images =np.expand_dims(test_images,-1)
model_name='proj_model.h5'
model = load_model(model_name)
t1 = timeit.default_timer()
#for x in range(100):
prediction=model.predict(test_images)
t2 = timeit.default_timer()
t = round(1000 * (t2 - t1), 2)/20
print(prediction)
prediction=np.argmax(prediction,axis=1)
print(prediction)
print(test_labels)
show_images_labels_predictions(test_images,test_labels,prediction,0,len(test_images),t)
