import os
import sys
pth = os.path.abspath(__file__)
nam = pth.split(os.sep)[-1]
abspath = pth.replace(nam, "")

import cv2
import numpy as np
from twodim_KalmanFilter import KalmanFilter
import imageio

def main():
    #Variable used to control the speed of reading the video
    ControlSpeedVar = 100  #Lowest: 1 - Highest:100

    HiSpeed = 100+10

    #Create KalmanFilter object KF
    #KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)
    KF = KalmanFilter(0.1, 1, 1, 1, 0.1,0.1)

    centersall = []
    with open(os.path.join(abspath, 'center.txt'), 'r') as obj:
        for i in obj.readlines():
            i = i.strip().split(" ")
            i = [float(i[0]), float(i[1])]
            centersall.append([[[i[0]], [i[1]]]])
    centersall = np.array(centersall)
    num= 0
    framesall = []
    while(True):
        # Detect object
        centers = centersall[num % len(centersall)]
        num += 1
        if num==len(centersall):
            break
        if num > len(centersall):
            num = num % len(centersall)
        
        frame = np.ones((360, 660-20, 3), dtype = np.uint8) * (2**8 - 1)
        # If centroids are detected then track them
        if (len(centers) > 0):

            # Draw the detected circle
            cv2.circle(frame, (int(centers[0][0][0]), int(centers[0][1][0])), 10, (0, 191, 255), 2)

            # Predict
            (x, y) = KF.predict()
            # Draw a rectangle as the predicted object position
            x, y = x[0], y[0]
            cv2.rectangle(frame, (int(x - 15), int(y - 15)), (int(x + 15), int(y + 15)), (255, 0, 0), 2)

            # Update
            (x1, y1) = KF.update(centers[0])

            # Draw a rectangle as the estimated object position
            x1, y1 = x1[0], y1[0]
            cv2.rectangle(frame, (int(x1 - 15), int(y1 - 15)), (int(x1 + 15), int(y1 + 15)), (0, 0, 255), 2)

            cv2.putText(frame, "Estimated Position", (int(x1 + 15), int(y1 + 10)), 0, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, "Predicted Position", (int(x + 15), int(y)), 0, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, "Measured Position", (int(centers[0][0][0] + 15), int(centers[0][1][0] - 15)), 0, 0.5, (0,191,255), 2)

        cv2.imshow('image', frame)
        framesall.append(frame)
        if cv2.waitKey(2) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

        cv2.waitKey(HiSpeed-ControlSpeedVar+1)

    with imageio.get_writer(os.path.join(abspath, r'gif.gif'), mode="I") as obj:
        for id, frame in enumerate(framesall):
            obj.append_data(frame)

if __name__ == "__main__":
    # execute main
    main()
