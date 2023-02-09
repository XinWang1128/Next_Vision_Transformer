
# Readme.md
# Workflow of Next Vision Transformer


#I just built a new docker image "vit" in tum node 6

$docker build -f ViT.dockerfile -t vit .

#I am in the step of setting up the docker env, and training the coco2017 data sets with my NVT model.

$ docker run -it --ipc=host --gpus all -v /home/xin:/source/Next-Vit -v /data/ssd/cv_data/:/source/data/ -w /source vit:latest

$ cd detection/

$ PORT=29501 bash dist_train.sh configs/mask_rcnn_nextvit_small_1x.py 1



