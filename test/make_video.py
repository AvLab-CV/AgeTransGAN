import cv2
import numpy as np
import os
import sys

def make_video(name, gp):

    ten_gp = ['0-2', '3-6', '7-9', '10-14', '15-19', '20-29', '30-39', '40-49', '50-69', '70+']
    four_gp = ['15-29', '30-39', '40-49', '50+']

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,50)
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2

    data_list = os.listdir('result/{}'.format(gp))
    data_list.sort()

    out = cv2.VideoWriter('save/{}.mp4'.format(name,gp),cv2.VideoWriter_fourcc(*'mp4v'), 10, (1024,1024))
    for j in range(2):
        if j == 1:
            data_list.reverse()
            four_gp.reverse()
            ten_gp.reverse()
        for i, path in enumerate(data_list):
            ind = i // 10
            text = four_gp[ind] if gp == 4 else ten_gp[ind]

            img = cv2.imread('result/{}/{}'.format(gp, path))
                
            cv2.putText(img, 'Age:{}'.format(text),
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    lineType)
            out.write(img)
    out.release()

def play_video(name):
    cap = cv2.VideoCapture('save/{}.mp4'.format(name))
    frame_counter = 0
    while(True):
        ret, frame = cap.read() 
        frame_counter += 1
        if frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            frame_counter = 0 #Or whatever as long as it is the same as next line
            cap = cv2.VideoCapture('save/{}.mp4'.format(name))
            cap.set(cv2.CAP_PROP_FPS, 1)

        cv2.imshow("Facial Age Transfer", frame)
        cv2.waitKey(20)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # make_video('20200702-16-50-59', 10)
    play_video(sys.argv[1])
