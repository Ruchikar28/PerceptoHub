import os
import cv2

base_dir = './data'
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

number_of_classes = 26
class_images = 400

cap = cv2.VideoCapture(0)
for j in range(number_of_classes):
    if not os.path.exists(os.path.join(base_dir, str(j))):
        os.makedirs(os.path.join(base_dir, str(j)))

    print('collecting data for class {}', format(j))

    done = False
    while(True):
        ret, frame = cap.read()
        cv2.putText(frame, 'press enter to capture ', (100, 50), cv2.FONT_HERSHEY_SIMPLEX,1.3,(0,255,0),3,cv2.LINE_AA)
        cv2.imshow('frame',frame)
        if cv2.waitKey(25) == ord('p'):
            break

    counter =0
    while counter< class_images:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('e'):
            print('Exiting')
            break
        cv2.imwrite(os.path.join(base_dir, str(j), '{}.jpg'.format(counter)), frame)    
    
        counter += 1

    if counter < class_images:
        print(f"Restarting capture for {j}.")
    else:
        print(f"Completed capture for {j}.")    


cap.release()
cv2.destroyAllWindows()        

