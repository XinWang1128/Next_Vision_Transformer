# -------------------------------
# author: Hao Li, hao.li@uni-heidelberg.de
# data: 22.04.2021
# -------------------------------

import os
import re
import math
import json
from osgeo import ogr
from copy import deepcopy
from shapely.geometry import Polygon
import geopandas as gpd


def pixel_coords_zoom_to_lat_lon(PixelX, PixelY, zoom):
    MapSize = 256 * math.pow(2, zoom)
    x = (PixelX / MapSize) - 0.5
    y = 0.5 - (PixelY / MapSize)
    lon = 360 * x
    lat = 90 - 360 * math.atan(math.exp(-y * 2 * math.pi)) / math.pi

    return lon, lat


def score_filter(min_score=10):
    new_pred = {}
    new_pred['type'] = predictions['type']
    new_pred['features'] = []
    for feature in predictions['features']:
        if feature['properties']['score'] >= min_score:
            new_pred['features'].append(feature)
    return new_pred

def parse_tile_name(name):
    zoom, TileX, TileY = [int(x) for x in name.split(".")]
    return TileX, TileY, zoom


def feature_coords_to_lat_lon(feature):
    TileX, TileY, zoom = parse_tile_name(feature['properties']['task_id'])
    PixelX = TileX * 256
    PixelY = TileY * 256
    coords = feature['geometry']['coordinates'][0]
    print(feature['properties']['task_id'])
    translated = [[PixelX + y, PixelY + x] for x, y in coords]
    transformed = [[i for i in pixel_coords_zoom_to_lat_lon(x, y, zoom)] for x, y in translated]

    return transformed


if __name__ == "__main__":
    # # ------------- put your input here--------------------
    root_path = "D:/Code/ohsome2label-workingrepo/ohsome2label-master/generate_TFRecords/ssl_small_90/predicton_results/"
    raw_predictions_path = root_path + "OD_prediction_SSL_two.geojson"
    processed_prediction_path = root_path + "transformed_OD_prediction_SSL_two.geojson"
    with open(raw_predictions_path) as f:
        predictions = json.load(f)
        # filter the score value larger than with 50
        predictions = score_filter()
    for feature in predictions['features']:
        feature['geometry']['coordinates'][0] = feature_coords_to_lat_lon(feature)
    with open(processed_prediction_path, "w") as f:
        json.dump(predictions, f)
