import cv2,numpy,os
import sys

#cascPath = sys.argv[1]

datasets='datasets'

sub_data='hritvik'
path=os.path.join(datasets, sub_data)
if not os.path.isdir(path):
    os.mkdir(path)

(width, height)=(100,100)

faceCascade = cv2.CascadeClassifier('face.xml')
count=1
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        #minSize=(30, 30),
        #flags=cv2.CV_HAAR_SCALE_IMAGE
    )
    #print("Found {0} faces!".format(len(faces)))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0, 0), 2)
        face=gray[x:x+w,y:y+h]
        face_resize=cv2.resize(face,(width,height))

    while count<30:
        cv2.imwrite('% s/% s.png'%(path,count), face_resize)
        count=count+1

    # Display the resulting frame
    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#print(count)
#cv2.imwrite('hritvik.png',img)
print("Found {0} faces!".format(len(faces)))
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
