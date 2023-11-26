import torch
import pandas as pd
import numpy as np
import math
import ultralytics.engine.results
from ultralytics import YOLO
from shapely.geometry import Polygon
from PIL import Image, ImageDraw
from sklearn.linear_model import LinearRegression

do_save = False

model_seg = YOLO('yolov8x-seg.pt')

colors = np.array([[0, 128, 0], [144, 238, 144], [255, 255, 0], [255, 165, 0], [255, 0, 0]])
ripeness = np.array([1, 2, 3, 4, 5])
maturity_model = LinearRegression()
maturity_model.fit(colors, ripeness)

def get_size(coords: list) -> float:
    polygon = Polygon(coords)
    return polygon.area

def get_percent(coords: list, box: ultralytics.engine.results.Boxes) -> float:
    box_size = int(box[2] * box[3])
    apple_size = get_size(coords)
    return apple_size / box_size

def get_maturity_from_rgb(rgb: list) -> float:
    maturity = maturity_model.predict(rgb)
    return maturity

def calc_maturity(filename: str, results: ultralytics.engine.results.Results, id: int) -> float:
    def create_polygon_mask(points, width, height):
        mask = Image.new('1', (width, height), 0)
        draw = ImageDraw.Draw(mask)
        draw.polygon(points, fill=1)
        return mask
    
    img = Image.open(filename)
    coords = results.masks[id].xy[0]
    mask = create_polygon_mask(coords, 270, 185)

    count = np.array(mask).sum()
    sum = (0, 0, 0)
    n, m = mask.size
    for i in range(n):
        for j in range(m):
            if not mask.getpixel((i, j)):
                continue
            pixel = img.getpixel((i, j))
            sum = [x + y for (x, y) in zip(sum, pixel)]
    avg_color = sum / count

    maturity = get_maturity_from_rgb([avg_color])
    print(avg_color, maturity)
    return maturity

def feature_fusion(percent: float, maturity: float) -> float:
    return (percent, maturity)

def calc_apple_percent(results: ultralytics.engine.results.Results, id: int) -> float:
    coords = results.masks[id].xy[0]
    box = results.boxes[id].xywh[0]
    percent = get_percent(coords=coords, box=box)
    percent = math.pow(percent, 1.5)
    return percent

def process_image(filename: str) -> float:
    results = model_seg(filename, save=do_save, save_txt=do_save, save_conf=do_save, device='cuda')[0]
    apple_ids = torch.nonzero(results.boxes.cls == 47)
    percent_list = []
    maturity_list = []
    for id in apple_ids:
        percent = calc_apple_percent(results, id)
        maturity = calc_maturity(filename, results, id)
        percent_list.append(percent)
        maturity_list.append(maturity.tolist())
    return (percent_list, maturity_list)


percent_list = []
maturity_list = []
for image_id in range(1, 201):
    new_features = process_image(f'data/Attachment1/{image_id}.jpg')
    percent_list.extend(new_features[0])
    maturity_list.extend(new_features[1])
pd.DataFrame({'percent': percent_list, 'maturity': maturity_list}).to_csv('p4-features.csv', index=False)
