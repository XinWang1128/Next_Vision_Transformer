from argparse import ArgumentParser
import os
from mmdet.apis import inference_detector, init_detector  #, show_result_pyplot
import cv2
import nextvit
import geopandas as gpd
from shapely.geometry import box
from shapely.geometry import box, mapping



def show_result_pyplot(model, img, result, score_thr=0.3, fig_size=(15, 10)):
    """Visualize the detection results on the image.
    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loadaed image.
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
    config_file = './configs/mask_rcnn_nextvit_large_1x.py'
    # 训练好的模型
    checkpoint_file = './work_dirs/mask_rcnn_nextvit_large_1x/epoch_12.pth'

    # model = init_detector(config_file, checkpoint_file)
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    # Create an empty GeoDataFrame to store the predicted bounding boxes
    gdf = gpd.GeoDataFrame(columns=['geometry'])
    # 图片路径
    img_dir = './data/test_sample/'
    # 检测后存放图片路径
    out_dir = './prediction_output_large_new_0.9/'

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

        # Extract the bounding box coordinates from the result
        bboxes = result[0]

        for bbox in bboxes:
            if len(bbox) >= 4:  # Check if the bbox has at least 4 values
                xmin, ymin, xmax, ymax = bbox[:4]  # Extract the first 4 values as xmin, ymin, xmax, ymax
                # Create a Shapely geometry object from the bounding box coordinates
                bbox_geometry = box(xmin, ymin, xmax, ymax)

                # Create a GeoDataFrame with the bounding box geometry
                bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_geometry])

                # Add the GeoDataFrame to the main GeoDataFrame
                gdf = gdf.append(bbox_gdf)
            else:
                # Handle the case when the bbox does not have enough values
                print("Invalid bbox:", bbox)
        img = show_result_pyplot(model, name, result, score_thr=0.9)
        cv2.imwrite("{}/{}.jpg".format(out_dir, test), img)

    # Save the GeoDataFrame to a GeoJSON file
    gdf.to_file('predicted_bboxes.geojson', driver='GeoJSON')

"""        # Extract the ground truth and predicted labels
        true_label = extract_true_label(test)  # Replace with your logic to get the ground truth label
        predicted_label = extract_predicted_label(result)  # Replace with your logic to get the predicted label"""
"""
        true_labels.append(true_label)
        predicted_labels.append(predicted_label)"""
            # Calculate the confusion matrix

if __name__ == '__main__':
    main()