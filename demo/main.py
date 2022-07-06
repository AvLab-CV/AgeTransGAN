import os
import argparse
from PIL import Image
import lm_process
import img_process
import age_transfer
from torch.autograd import Variable
import torch
from torchvision import transforms
import numpy as np
import cv2
from datetime import datetime
from make_video import make_video, play_video


def main(args):
    LMP = lm_process.LandmarkProcessing()
    IMP = img_process.ImageProcessing()
    model = age_transfer.Model(args)
    
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)])
    flag = False
    while (True):
        try:
            if flag == False:
                flag = True
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                print('\nPress c to capture face, and ESC to quit')
            ret, frame = cap.read()
            cv2.imshow('Frame', frame)
            key = cv2.waitKey(1)
            if key == 99: 
                save_name = datetime.now().strftime('%Y%m%d-%H-%M-%S')
                capture_img = frame
                print('Landmark detection')
                lm = LMP.detector(capture_img)
                print('Done')
                cap.release()
                cv2.destroyAllWindows()
                print('------------------')
                cropped = IMP.crop(capture_img, lm)
                cv2.imwrite('crop/{}.png'.format(save_name), cropped)
                cropped = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                cropped = transform(cropped)
                print('Facial Age Transformation')
                model.generate_image(cropped, args.group)
                print('\nDone')
                print('------------------')
                print('Generate Video')
                make_video(save_name, args.group)
                print('Done')
                print('------------------')
                print('Press q to quit')
                play_video(save_name)
                flag = False
                continue
            elif key == 27: 
                print('exit')
                break
        except:
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, default=1024, help='cropping size')
    parser.add_argument('--group', type=int, default=10, help='initial Number of ID [default: 188]')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--snapshot', type=str, default='./snapshot/140000.pt', help='filename of model snapshot(snapshot/{Single or Multiple}/{date}/{epoch}) [default: None]')

    args = parser.parse_args()
    main(args)
