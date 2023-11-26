import os
import matplotlib.pyplot as plt
from ultralytics import YOLO

model = YOLO("yolov8x.pt")
xy = []

# Predict an image and output coordinates of apples
def predict(filename: str) -> list:
    if not os.path.exists(filename):
        return []
    results = model.predict(filename, conf=0.1, save=True, save_txt=True, augment=True, retina_masks=True)
    xy = results[0].boxes.xyxy.cpu().tolist()
    xy = [(i[0], 185 - i[3]) for i in xy]
    return xy

for i in range(1, 201):
    for cood in predict(f'Attachment1/{i}.jpg'):
        xy.append(cood)

# Draw plots
fig, ax = plt.subplots()
ax.invert_yaxis()
for (x, y) in xy:
    ax.plot(x, y, 'ro') 
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.savefig('test.jpg')