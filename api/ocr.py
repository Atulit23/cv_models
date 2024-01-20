from paddleocr import PaddleOCR,draw_ocr
import os
import cv2
import matplotlib.pyplot as plt

ocr = PaddleOCR(use_angle_cls=True, lang='en')

img_path = 'api/default_1280-720-screenshot.webp'
result = ocr.ocr(img_path)

boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]

#   im_show = draw_ocr(image, boxes, txts, scores, font_path='api/simfang.ttf')
  
#   cv2.imwrite(save_path, im_show)
 
#   img = cv2.cvtColor(im_show, cv2.COLOR_BGR2RGB)
#   plt.imshow(img)

# out_path = 'C:/Users/atuli/OneDrive/Desktop/darkine_new/results'
# save_ocr(img_path, out_path, result)

print(boxes)
print(txts)

# the order of the four points in each detection box is upper left, upper right, lower right, and lower left.
# 0 ---> x1, y1
# 1 ---> x2, y2
