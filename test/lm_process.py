import face_alignment
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import datetime
import math
import copy
import util


class LandmarkProcessing():
    def __init__(self):
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

    def detector(self, img):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        preds = self.fa.get_landmarks(np.asarray(img))

        lms = np.array(preds).reshape(-1,136)

        df = pd.DataFrame(lms)
        return self.filter(df)

    def filter(self, df):
        multi_lm = len(df)
        lm_data = self.auto_filter(df) if multi_lm > 1 else df
        return lm_data.values

    def auto_filter(self, df):
        covs = []
        points_dis = []
        for idx in range(len(df)):
            distance = []
            points = [0,16,17,21,22,26,27,30,31,35,36,41,42,47,48,67]
            landmarks = df.iloc[idx, :].values
            landmarks = landmarks.astype('float').reshape(-1, 2)
            for i in range(8):
                for j in range(points[i*2],points[i*2+1]):
                    dis = math.sqrt((landmarks[j+1, 0]-landmarks[j, 0])**2 + (landmarks[j+1, 1] - landmarks[j, 1])**2)
                    distance.append(dis)
            cov = np.std(distance)/np.mean(distance)
            covs.append(cov)
            lm_temp = copy.deepcopy(landmarks)
            horizontal = math.sqrt((lm_temp[16, 0] - lm_temp[0, 0])**2 + (lm_temp[16, 1] - lm_temp[0, 1])**2)
            vertical = math.sqrt((lm_temp[28, 0] - lm_temp[9, 0])**2 + (lm_temp[28, 1] - lm_temp[9, 1])**2)
            points_dis.append(horizontal + vertical)

        covs = np.array(covs)
        points_dis = np.array(points_dis)
        points_dis[covs<0.32] = 0
        points_dis[covs>0.74] = 0
        save_idx = np.argmax(points_dis)
        return df.iloc[save_idx]