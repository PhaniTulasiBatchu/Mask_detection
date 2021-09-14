import cv2
import numpy as np
from keras.models import load_model

model=load_model(r'C:\Users\ELCOT\Desktop\july 10th\model-008.model')

from playsound import playsound

face_clsfr=cv2.CascadeClassifier(r'C:\Users\ELCOT\AppData\Local\Programs\Python\Python37\Haar_cascade_frontalface_default.xml')



labels_dict={0:'NO MASK',1:'MASK'}
color_dict={0:(0,0,255),1:(0,255,0)}

imgg=cv2.imread(r'C:\Users\ELCOT\Downloads\images\finlly-we.jpeg')


new_width = int(imgg.shape[1] * 0.5)
new_height = int(imgg.shape[0] * 0.5)

newsize = (new_width, new_height)

im= cv2.resize(imgg, newsize)
cv2.imshow('original',im)

img=cv2.cvtColor(im,cv2.COLOR_BGR2BGRA)


gray=cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)
faces=face_clsfr.detectMultiScale(gray,1.3,5)
print('no. of faces found=',len(faces))

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
        playsound(r'C:\Users\ELCOT\Downloads\music\speech.mp3')
        
    cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
    cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
    cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(225,255,255),2)
    cv2.rectangle(img,(490,580),(650,650),(250,250,250),-1)
    cv2.putText(img, "PPL=" + str(faces.shape[0]), (500,610),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2)
    cv2.imshow('after detection',img)
    cv2.waitKey(0)
    
cv2.destroyAllWindows()

