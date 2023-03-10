
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

$docker build -f ViT.dockerfile -t vit .

#I am in the step of setting up the docker env, and training the coco2017 data sets with my NVT model.

$ docker run -it --ipc=host --gpus all -v /home/xin:/source/Next-Vit -v /data/ssd/cv_data/:/source/data/ -w /source vit:latest

$ cd detection/

$ PORT=29501 bash dist_train.sh configs/mask_rcnn_nextvit_small_1x.py 1


#change the data to my german data, but there is an error about the the nummer of class, so we should change coco.py(fixed by add "classes = ('wastewater', ),")


# To do

1. check the training accrucy
#hard to see, because it's not the detection accuracy of the WWTP class.
![result chat](https://github.com/XinWang1128/Next_Vision_Transformer/blob/master/98d7b963-95c3-471b-be51-7d886cbcae4d.png)

2. save the model and make a prediction

*** figure it out, which size or type of input should we use


save the model from training step(https://github.com/GIScience/ohsome2label/tree/master/walkthrough)
#accroding to reading the "export_inference_graph.py"(https://github.com/tensorflow/models/blob/master/research/object_detection/export_inference_graph.py), the saving model function is based on TF. So the problems are if we could use the pytorch image to run the TF based functions.

3. make a prediction with our stuttgart test area (if we should split it into train and test)
4. get the results and visualization
5. calculate the accuracy binary classification by ourself, to make a confusion matirx(recall, prediction, recall)


References:
Next-ViT: https://github.com/bytedance/Next-ViT




