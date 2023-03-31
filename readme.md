
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

2. save the model and make a prediction

$ PORT=29501 bash dist_test.sh configs/mask_rcnn_nextvit_small_1x.py ../checkpoints/mask_rcnn_1x_nextvit_small.pth 1 --eval bbox

after change the classes of coco.py(from 80 classes to 2), we have made a valuation about our model, and it shows a very bad result. So we should take a look at the input data and the hyperparameters again.Or, run the training again.
![WhatsApp Image 2023-02-13 at 10 10 56](https://user-images.githubusercontent.com/87394529/224976140-014632d5-aed1-48d7-ba0e-9ca1c669921f.jpg)(This figure is a result of the object detection based on coco dataset and mask_rcnn_nextvit_small_1x.py)
![result chat](https://github.com/XinWang1128/Next_Vision_Transformer/blob/master/98d7b963-95c3-471b-be51-7d886cbcae4d.png)
(Here is our object detection accuracy,but the accuracy is very low based on next_rcnn_small_1x.py)



(Here is our object detection accuracy with next_rcnn_large_3x.py, the accuracy becomes a liitle bit better but not that better at all. I guess, it is probably the bbox of the input are very small, and the ViT can only learn from the big bbox or objects)

Because of the low prediction accuracy, we would like to check, if our the label of our train_annotation is right or not.(Atfer checking the bbox index, we find the x, y values perhaps wrong, we should use x value as our y, and use our y value as our x)
The comparation between the two different bbox in our OSM data:

With the old bbox:

![image](https://user-images.githubusercontent.com/87394529/225911088-cba04cf2-df8b-4951-aab1-a89b2ba2043c.png)
![image](https://user-images.githubusercontent.com/87394529/226318690-bf117352-47f1-4241-a593-5ba797a018bd.png)


With fixed the bbox:

![image](https://user-images.githubusercontent.com/87394529/225911257-824213a7-f09f-492f-915b-2adaed9e397d.png)
![image](https://user-images.githubusercontent.com/87394529/226318776-1467d2b0-3d3b-4c25-8703-fc247c2df955.png)



But the mask of coco works pretty fine:

The  bbox in COCO data:

![image](https://user-images.githubusercontent.com/87394529/225911454-4ca671af-22bb-452d-b231-5390d685ad27.png)

But if we use all the bbox to check the coco dataset, it works:

![image](https://user-images.githubusercontent.com/87394529/226360667-cfdf3fa5-4322-4832-b96d-79ba4685ecfa.png)
![image](https://user-images.githubusercontent.com/87394529/226360704-21094e88-bb5b-4114-aa92-7e75d9863bd1.png)
![image](https://user-images.githubusercontent.com/87394529/226360839-be6aea10-2fc5-4975-bdfd-9f69f18bee33.png)
![image](https://user-images.githubusercontent.com/87394529/226360795-14a05282-f18e-44e0-bf1d-e93d30c76285.png)


3. make a prediction with our stuttgart test area (if we should split it into train and test)
The index:

I. batch_size and LR are proportional

If you have only one gpu,samples_per_gpu=8,batch_size=8,your new learning rate lr = 0.005





4. get the results and visualization

The results like:

![176876645028](https://user-images.githubusercontent.com/87394529/226669488-96bc8c2c-2583-4c09-818c-440956f6ba47.jpg)
![176878445140](https://user-images.githubusercontent.com/87394529/226669522-8ef0327b-57d8-4ede-a86c-2317c00934ec.jpg)


Validation data:

![image](https://user-images.githubusercontent.com/87394529/227224906-6e94bfff-590e-45b0-be17-9d84c01a4f04.png)
![image](https://user-images.githubusercontent.com/87394529/227225026-a4fdad82-d275-48f0-991a-3ce8488a94e8.png)
![image](https://user-images.githubusercontent.com/87394529/227225093-25beff8e-816d-434c-b5c2-00f151a9a557.png)
![image](https://user-images.githubusercontent.com/87394529/227225172-84f24f6a-263a-447c-8365-6ed98fcb8a47.png)
![image](https://user-images.githubusercontent.com/87394529/227225253-7027dd27-2939-4091-90d0-d7315e70cb1b.png)
![image](https://user-images.githubusercontent.com/87394529/227225388-49e70300-7e34-41f5-abdb-1ed0bc7317ec.png)
![image](https://user-images.githubusercontent.com/87394529/227225439-0235b28a-ca28-4034-ba73-9908265c8c69.png)
![image](https://user-images.githubusercontent.com/87394529/227225499-201a7b0a-47b5-4aef-b60d-daeffc4b2c40.png)
![image](https://user-images.githubusercontent.com/87394529/227225555-3f1ec1c0-4ac0-4529-bdc7-2c8d028b5f87.png)



5. calculate the accuracy binary classification by ourself, to make a confusion matirx(recall, prediction, recall)


References:
Next-ViT: https://github.com/bytedance/Next-ViT




