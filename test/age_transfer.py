import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import Generator
import util
from torchvision import utils
import numpy as np
import matplotlib.pyplot as plt
import cv2

class Model():
    def __init__(self, args):
        self.group = args.group
        self.batch = args.batch_size
        self.G_model = self.load_model(args)

    def load_model(self, args):
        if args.snapshot is None:
            print("Sorry, please set snapshot path while generate")
            exit()
        else:
            G = Generator(1024, 512, 8, self.group, channel_multiplier=2)
            checkpoint = torch.load(args.snapshot)
            G.load_state_dict(checkpoint['g_ema'])

        G.cuda()
        G.eval()

        return G

    def generate_image(self, batch_image, group):

        age_code = torch.LongTensor(np.arange(group-1)).repeat(1,10)[0].sort()[0]
        one_hot = util.one_hot(age_code, group)
        last = util.one_hot(torch.LongTensor([group-1]), group)
        for i in range(len(one_hot)):
            re = i // 10
            qu = i % 10
            one_hot[i][re] = one_hot[i][re] - (qu * 0.1)
            one_hot[i][re+1] = one_hot[i][re+1] + (qu * 0.1)
        one_hot = torch.cat([one_hot, last])
        total_data = len(one_hot)
        
        #print(one_hot)
        batch_image = np.repeat(batch_image.reshape((1,3,1024,1024)), total_data, axis=0)
        batch_image = torch.FloatTensor(batch_image)

        batch_image = torch.split(batch_image, self.batch)
        batch_age_code = torch.split(one_hot, self.batch)
        
        count = 0
        total = len(age_code)
        with torch.no_grad():
            for i in range(len(batch_image)):
                mini_age_code = Variable(batch_age_code[i].cuda())
                mini_img = Variable(batch_image[i].cuda())
                generated = self.G_model(mini_img, mini_age_code)
                for j in range(len(mini_img)):
                    utils.save_image(generated[0][j],
                    'result/{}/{}.png'.format(group, str(count).zfill(3)), nrow=1, normalize=True, range=(-1, 1))
                    print("\rProcessing {:.2f}%".format(100*count/total), end="", flush=True)
                    count += 1

            

    
