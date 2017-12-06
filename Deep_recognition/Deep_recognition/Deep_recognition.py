import face_recognition
from os import listdir
from os.path import isfile, join
import sys
import time
import cv2

def getReferences(dir):
    subdirs=listdir(dir) 
    names=[];
    emb=[]
    for subdir in (subdirs):
        name=subdir
        subdir=join(dir,subdir)
        files=[f for f in listdir(subdir) if isfile(join(subdir,f))]
        for file in (files):
            file=join(subdir,file);
            image = face_recognition.load_image_file(file)
            encodings=face_recognition.face_encodings(image)
            if (encodings):
                names.append(name)
                emb.append(encodings[0])
    return emb,names

def proccessVideo(dir,dist):
    [references,names]=getReferences(dir+"reference")
    dir=dir+"test"
    subdirs=listdir(dir) 
    TP=0
    FP=0
    FN=0
    TN=0
    total=0
    for subdir in (subdirs):
        name=subdir
        subdir=join(dir,subdir)
        files=[f for f in listdir(subdir) if isfile(join(subdir,f))]
        for file in (files):
            file=join(subdir,file);
            videoCapture=cv2.VideoCapture(file)
            if (not videoCapture.isOpened()):
                continue
            embeddings=[]
            ret,frame=videoCapture.read()
            while(ret):
                image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                encodings=face_recognition.face_encodings(image)
                if (encodings):
                    embeddings.append(encodings[0])
                ret,frame=videoCapture.read()
            for n, tst in enumerate(embeddings):
                face_distances = face_recognition.face_distance(references, tst)
                for i, face_distance in enumerate(face_distances):
                    if (face_distance<dist):
                        if (name==names[i]):
                            TP+=1      
                        else:
                            FP+=1
                    else:
                        if (name==names[i]):
                            FN+=1      
                        else:
                            TN+=1
                    total=total+1
            print("TP=",TP," FP=",FP," FN=",FN," TN=",TN," Total=",total)

def main(args,distance):
    proccessVideo("C:/Users/Veronika/Desktop/video/",distance)
    return 0
    subdirs=listdir(args) 
    features=[];
    names=[];
    for subdir in (subdirs):
        name=subdir
        subdir=join(args,subdir)
        files=[f for f in listdir(subdir) if isfile(join(subdir,f))]

        for file in (files):
            file=join(subdir,file);
            image = face_recognition.load_image_file(file)
            start=time.time()
            align=face_recognition.face_landmarks(image)
            time1=time.time()-start
            start=time.time()
            encodings=face_recognition.face_encodings(image)
            stop=time.time()
            time2=stop-start
            print("Time = ",time2-time1)
            if (encodings):
                features.append(encodings[0])
                names.append(name)
   
    TP=0
    FP=0
    TN=0
    FN=0
    total=0;
    for n, tst in enumerate(features):
        face_distances = face_recognition.face_distance(features, tst)
        for i, face_distance in enumerate(face_distances):
            if (i<=n):
                continue
            if (face_distance<distance):
                if (names[n]==names[i]):
                    TP+=1      
                else:
                    FP+=1
                    print("FP names ",names[n]," ",names[i])                 
            else:
                if (names[n]==names[i]):
                    FN+=1   
                    print("FN names ",names[n]," ",names[i])            
                else:
                    TN+=1
                    
            total=total+1;
    print("TP=",TP," FP=",FP," FN=",FN," TN=",TN," Total=",total)
    input("Press Enter to continue ...")
    return 0

if __name__=="__main__":
    sys.exit(int(main("C:/Users/Veronika/Desktop/facenet-master/src/align/datasets/lfw/raw",0.55)) or 0)
