#!/bin/bash
#source ~/virtualenv/pt1.4_tf2.1/bin/activate
python main.py --img_size 1024 --group 10 --batch_size 16 --snapshot ./snapshot/ffhq_10group_820k.pt
deactivate
