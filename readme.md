
# Readme.md
# Workflow of Next Vision Transformer


#Hi, me.


#I am in the step of setting up the docker env, and training the coco2017 data sets with my NVT model.

$ docker run -it --rm --ipc=host --name=xin -p 8889:8888 -v /home/xin/:/source/Next-ViT/ -v /data/ssd/cv_data/:/source/data/ -w /source pytorch/pytorch:latest

$ cd detection/
$ PORT=29501 bash dist_train.sh configs/mask_rcnn_nextvit_small_1x.py 1


