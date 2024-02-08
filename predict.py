import cv2
from ultralytics import YOLO

image  = cv2.imread("/home/aditya/Desktop/photos_waste/IMG_20240105_154106.jpg")


model_path = "/home/aditya/Desktop/data/runs/detect/train/weights/last.pt"

model = YOLO(model_path)

threshold = 0.5

r_img = cv2.resize(image, (500, 500))

results = model(r_img)

area = 0

def area_calc(x1, y1, x2, y2):
    x1, y1, x2, y2 = x1, y1, x2, y2
    length = abs(x1 - x2)
    width = abs(y1 - y2)
    return length * width

import cv2

# Assuming `results` is a list of objects
for result in results:
    # Access the 'boxes' attribute of each result item
    boxes = result.boxes
    # Convert the boxes to a list
    boxes_list = boxes.data.tolist()
    print(boxes_list, "\n\n")
    print(len(boxes_list))
    
    for o in boxes_list:
        x1, y1, x2, y2, score, class_id = o
        pred_img = cv2.rectangle(r_img, (int(x1), int(y1)), (int(x2), int(y2)) ,(0, 255, 0), 2)
        x = area_calc(x1, y1, x2, y2)
        area = area + x;

print("\n\nArea of waste detected is ", area, "unit sq", "\n\n")
print("Area of the image is ", 500*500, "unit sq")

print("The Percentage of Waste detected in the image is ", round((area / (500*500)) * 100), "%")

cv2.imshow("Predicted Waste Products", r_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
