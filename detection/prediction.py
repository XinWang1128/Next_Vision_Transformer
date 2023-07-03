from argparse import ArgumentParser
import os
from mmdet.apis import inference_detector, init_detector  #, show_result_pyplot
import cv2
import nextvit
import numpy as np
from sklearn.metrics import confusion_matrix


def show_result_pyplot(model, img, result, score_thr=0.3, fig_size=(15, 10)):
    """Visualize the detection results on the image.
    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        fig_size (tuple): Figure size of the pyplot figure.
    """
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(img, result, score_thr=score_thr, show=False)
    return img
    # plt.figure(figsize=fig_size)
    # plt.imshow(mmcv.bgr2rgb(img))
    # plt.show()


def main():
    # config文件
    config_file = './configs/mask_rcnn_nextvit_base_1x.py'
    # 训练好的模型
    checkpoint_file = './work_dirs/mask_rcnn_nextvit_base_1x/epoch_12.pth'

    # model = init_detector(config_file, checkpoint_file)
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    # 图片路径
    img_dir = './data/test_sample/'
    # 检测后存放图片路径
    out_dir = './prediction_output_base_new/'

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # 测试集的图片名称txt
    test_path = './test_list.txt'
    fp = open(test_path, 'r')
    test_list = fp.readlines()
    true_labels = []
    predicted_labels = []

    count = 0
    imgs = []
    for test in test_list:
        test = test.replace('\n', '')
        test = test.split('.')[0]
        name = img_dir + test + '.jpg'
        count += 1
        print('model is processing the {}/{} images.'.format(count, len(test_list)))
        # result = inference_detector(model, name)
        # model = init_detector(config_file, checkpoint_file, device='cuda:0')
        result = inference_detector(model, name)
        img = show_result_pyplot(model, name, result, score_thr=0.8)
        cv2.imwrite("{}/{}.jpg".format(out_dir, test), img)

"""        # Extract the ground truth and predicted labels
        true_label = extract_true_label(test)  # Replace with your logic to get the ground truth label
        predicted_label = extract_predicted_label(result)  # Replace with your logic to get the predicted label"""
"""
        true_labels.append(true_label)
        predicted_labels.append(predicted_label)"""


            # Calculate the confusion matrix



if __name__ == '__main__':
    main()