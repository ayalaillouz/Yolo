import cv2

# מודל coco
coco = cv2.dnn.readNet('yolov5s.weights', 'data/coco128.yaml')

# מודל מותאם אישית
custom = cv2.dnn.readNet('custom.weights', 'data/custom_data.yaml')

# תמונה
image = cv2.imread('../27.jpg')

# זיהוי אובייקטים
detections = coco.detectMultiScale(image)
custom_detections = custom.detectMultiScale(image)

# איחוד התוצאות
detections += custom_detections

# חיפוש אובייקטים
for detection in detections:
    if detection['class'] == 'person':
        print('מצאתי אדם!')