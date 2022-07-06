#!/bin/bash
#source ~/virtualenv/pt1.4_tf2.1/bin/activate
python main.py --img_size 1024 --group 4 --batch_size 16 --snapshot ./snapshot/ffhq_4group_750k.pt --file img/1.jpg
deactivate
