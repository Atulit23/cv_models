from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

model = YOLO('api/all_elements.pt')
img = cv2.imread('api/Screenshot 2024-01-19 000410.png')

classes_ = {0: 'Button', 1: 'Edit Text', 2: 'Header Bar', 3: 'Image Button', 4: 'Image View', 5: 'Text Button', 6: 'Text View'}

results = model.predict(source=img, conf = 0.5)

# results = model.predict('api/default_1280-720-screenshot.webp', confidence=40, overlap=30).json()
boxes = results[0].boxes.xyxy.tolist()
classes = results[0].boxes.cls.tolist()
names = results[0].names
confidences = results[0].boxes.conf.tolist()

print(boxes)
print(classes)
# print(confidences)

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

plot_img_bbox(img, boxes)