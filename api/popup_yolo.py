# popup_det.pt
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

model = YOLO('api/popup_det.pt')
img = cv2.imread('api/default_1280-720-screenshot.webp')

classes_ = {0: 'noti', 1: 'pop'}

results = model.predict(source=img, conf = 0.8)

# results = model.predict('api/default_1280-720-screenshot.webp', confidence=40, overlap=30).json()
boxes = results[0].boxes.xyxy.tolist()
classes = results[0].boxes.cls.tolist()
names = results[0].names
confidences = results[0].boxes.conf.tolist()

print(boxes)
print(classes)
print(names)
print(confidences)

# Iterate through the results
for box, cls, conf in zip(boxes, classes, confidences):
    x1, y1, x2, y2 = box
    confidence = conf
    detected_class = cls
    name = names[int(cls)]

def plot_img_bbox(img, target):
    fig, a = plt.subplots(1,1)
    fig.set_size_inches(10, 10)
    a.imshow(img)
    for i, box in enumerate(target):
        #print(target['boxes'])
        x, y, width, height  = box[0], box[1], box[2]-box[0], box[3]-box[1]
#         if arr[target['labels'][i]] == 'ad':
        rect = patches.Rectangle((x, y),
                                     width, height,
                                     linewidth = 2,
                                     edgecolor = 'r',
                                     facecolor = 'none')
        a.text(x, y-20, classes_[classes[i]], color='b', verticalalignment='top')

        a.add_patch(rect)
    plt.show()

# if length of boxes is zero that means no deceptive popups were found
plot_img_bbox(img, boxes)
