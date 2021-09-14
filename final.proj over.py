import cv2
import os
import numpy as np

data_path=r'C:\Users\ELCOT\Downloads\data'

categories=os.listdir(data_path)

labels=[i for i in range(len(categories))]
label_dict=dict(zip(categories,labels))

print(label_dict)
print(categories)
print(labels)

data=[]
target=[]


for category in categories:
    folder_path=os.path.join(data_path,category)
    img_names=os.listdir(folder_path)
       
    for img_name in img_names:
        img_path=os.path.join(folder_path,img_name)
        img=cv2.imread(img_path)

        try:
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)          
            
            resized=cv2.resize(gray,(100,100))
            
            data.append(resized)
            target.append(label_dict[category])
            

        except Exception as e:
            print('Exception:',e)
            #if any exception rasied, the exception will be printed here. And pass to the next image
           

data=np.array(data)/255.0
data=np.reshape(data,(data.shape[0],100,100,1))
target=np.array(target)

from keras.utils import np_utils
target=np_utils.to_categorical(target)


from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint

model=Sequential()

model.add(Conv2D(200,(3,3),input_shape=data.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#The first CNN layer followed by Relu and MaxPooling layers

model.add(Conv2D(100,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#The second convolution layer followed by Relu and MaxPooling layers

model.add(Flatten())
model.add(Dropout(0.5))
#Flatten layer to stack the output convolutions from second convolution layer
model.add(Dense(50,activation='relu'))
#Dense layer of 64 neurons
model.add(Dense(2,activation='softmax'))
#The Final layer with two outputs for two categories

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

from sklearn.model_selection import train_test_split

train_data,test_data,train_target,test_target=train_test_split(data,target,test_size=0.1)
checkpoint = ModelCheckpoint('model-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
history=model.fit(train_data,train_target,epochs=20,callbacks=[checkpoint],validation_split=0.2)

print(model.evaluate(test_data,test_target))


#mask detection

from keras.models import load_model

model=load_model(r'C:\Users\ELCOT\Desktop\july 10th\model-008.model')

from playsound import playsound

face_clsfr=cv2.CascadeClassifier(r'C:\Users\ELCOT\AppData\Local\Programs\Python\Python37\Haar_cascade_frontalface_default.xml')

source=cv2.VideoCapture(0)

labels_dict={0:'NO MASK',1:'MASK'}
color_dict={0:(0,0,255),1:(0,255,0)}

while(True):

    ret,imgg=source.read()
    
    new_width = int(imgg.shape[1] * 0.5)
    new_height = int(imgg.shape[0] * 0.5)

    newsize = (new_width, new_height)

    im= cv2.resize(imgg, newsize)
    cv2.imshow('original',im)

    img=cv2.cvtColor(im,cv2.COLOR_BGR2BGRA)


    gray=cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)
    faces=face_clsfr.detectMultiScale(gray,1.3,5)
    print('no. of faces found=',len(faces)) 

    m=0
    
    for (x,y,w,h) in faces:
        face_img=gray[y:y+w,x:x+w]
        resized=cv2.resize(face_img,(100,100))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,100,100,1))
        result=model.predict(reshaped)
        label=np.argmax(result,axis=1)[0]

        if(label==0):
            playsound(r'C:\Users\ELCOT\Downloads\music\crazy.mp3')
        else:
            m=m+1
            playsound(r'C:\Users\ELCOT\Downloads\music\speech.mp3')
        
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(225,255,255),2)
        cv2.rectangle(img,(490,580),(650,650),(250,250,250),-1)
        cv2.putText(img, "ppl inside=" + str(faces.shape[0]), (480,620),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2)
        cv2.imshow('after detection',img)
        cv2.waitKey(1)
        
        print(m)
        
        if(m>2):
            playsound(r'C:\Users\ELCOT\Downloads\music\speech (2).mp3')
            cv2.imshow('thats it',img)
            cv2.waitKey(0)
            break;
        else:
            continue;
cv2.destroyAllWindows()
sourse.release()

