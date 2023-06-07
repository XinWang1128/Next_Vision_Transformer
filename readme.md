
# Readme.md
# Workflow of Next Vision Transformer

# Ohsome2label(Data preparing)

1. Check the ohsome2label API useful or not, use $ ohsome2label —help

2. Change the save direction and some details of output:
  name: WWTP
  workspace: ./BW_WWTP
  project_time: 2020-05-18
  task: object detection

3. Check the size of bboxes between output and the prepared one

4. Use the input like:
    - {'label': 'wastewater', 'key': 'landuse', 'value': 'basin'}


$ ohsome2label --config  C:\Users\24634\PycharmProjects\pythonProject\ohsome2label\config\config.yaml vector
II: change the .geojson file
$ ohsome2label label
$ ohsome2label image
$ ohsome2label visualize -n 10

#to split the train and test data from ohsome2label
python tf_record_from_coco.py --label_input=D:\MA\Data\Ohsome\ohsome2label\STUT_WWTP --train_rd_path=D:\MA\Data\Ohsome\ohsome2label\STUT_WWTP\train.json --valid_rd_path=D:\MA\Data\Ohsome\ohsome2label\STUT_WWTP\valid.json



# Docker
#I just built a new docker image "vit" in tum node 6

$ docker build -f ViT.dockerfile -t vit .

#I am in the step of setting up the docker env, and training the coco2017 data sets with my NVT model.

$ sudo docker run -it --ipc=host --gpus all -v /home/dev/Next-ViT:/source/Next-Vit -v /usr/local/cuda-11.8:/usr/local/cuda-11.8 -w /source/Next-Vit/detection nextvit1:latest

$ cd detection/

$ PORT=29501 bash dist_train.sh configs/mask_rcnn_nextvit_small_1x.py 1

$ PORT=29501 bash dist_train.sh configs/mask_rcnn_nextvit_base_1x.py 1

$ PORT=29501 bash dist_train.sh configs/mask_rcnn_nextvit_large_1x.py 1


#change the data to my german data, but there is an error about the the nummer of class, so we should change coco.py(fixed by add "classes = ('wastewater', ),")

# Work flow
1. Devided the total data sets to 2 big parts, Training samples and Test samples.

2. Split the training samples into training data and validation data

.jpg

3. Fit the training samples to our 3 inter-structures Next-ViT(small,base,large)
There are a comparation between these training results:

* small:



* base:



* large:



4. save the model and make a prediction
Perpare a test_samples folder, which have test images and negative images in this folder. And we need a txt file to get all a name list of all the images'names.
Use the prediction.py to make a prediction with the above images.
The comparations between the different sizes of Next-ViT



# To do

1.  calculate the accuracy binary classification by ourself, to make a confusion matirx(recall, prediction, recall)












References:
Next-ViT: https://github.com/bytedance/Next-ViT




