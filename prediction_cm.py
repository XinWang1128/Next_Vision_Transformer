from argparse import ArgumentParser
import os
from mmdet.apis import inference_detector, init_detector  #, show_result_pyplot
import cv2
import nextvit
import geopandas as gpd
from shapely.geometry import box

def save_predicted_bbox_as_raster(model, img_path, out_path, score_thr=0.3):
    result = inference_detector(model, img_path)
    bboxes = result[0]  # Assuming the bounding boxes are in the first element of the result tuple
    img = cv2.imread(img_path)

    bbox_gdf = gpd.GeoDataFrame()
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox  # Assuming bbox format as [xmin, ymin, xmax, ymax]
        polygon = box(xmin, ymin, xmax, ymax)
        bbox_gdf = bbox_gdf.append({'geometry': polygon}, ignore_index=True)

    bbox_gdf.to_file(out_path)

def main():
    # Configuration
    config_file = './configs/mask_rcnn_nextvit_large_1x.py'
    checkpoint_file = './work_dirs/mask_rcnn_nextvit_large_1x/epoch_12.pth'
    img_dir = './data/test_sample/'
    out_dir = './prediction_output_large_new/'

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Model initialization
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    # Process each image
    test_path = './test_list.txt'
    with open(test_path, 'r') as fp:
        test_list = fp.readlines()

    for idx, test in enumerate(test_list):
        test = test.strip()
        img_path = os.path.join(img_dir, test)
        out_path = os.path.join(out_dir, f"{test}.shp")

        print(f"Processing image {idx+1}/{len(test_list)}: {img_path}")
        save_predicted_bbox_as_raster(model, img_path, out_path, score_thr=0.8)

if __name__ == '__main__':
    main()
