from ultralytics import YOLO            #model
import numpy as np                      #computing
import matplotlib.pyplot as plt         #image output
import cv2 as cv                        #image processing
from skimage import io                  #image reader
import easyocr                          #text detection


#load models
#base_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('license_plate_detector.pt')

#trains model with custom made labeled data, higher epochs means more training
#numOfEpochs = 30
#license_plate_detector.train(data="data\data.yaml", epochs=numOfEpochs)


# specify easyOCR reading language and implements it as read_license_plate
reader = easyocr.Reader(['en'])
def read_license_plate(license_plate_crop, img):
    scores = 0
    detections = reader.readtext(license_plate_crop)

    width = img.shape[1]
    height = img.shape[0]
    
    #if there are no text detections
    if detections == [] :
        return None, None

    rectangle_size = license_plate_crop.shape[0]*license_plate_crop.shape[1]

    plate = [] 

    #checks through text detections to find relevent plate number based on size
    for result in detections:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        
        if length*height / rectangle_size > 0.17:
            bbox, text, score = result
            text = result[1]
            text = text.upper()
            scores += score
            plate.append(text)
    
    if len(plate) != 0 : 
        return " ".join(plate), scores/len(plate)
    else :
        return " ".join(plate), 0



# Load an image to test
image = io.imread(r"data\test\testImage1.jpg")
    
results = license_plate_detector.predict(image)  #runs the plate detector
result = results[0]
detectedCount = 0  #count for # of plates found 
x1, x2, y1, y2 = 0, 0 , 0, 0   #coordinates of plates
output = []   #stores output for matches

for box in result.boxes:   #loops for every match
    x1, y1, x2, y2 = [ round(x) for x in box.xyxy[0].tolist() ]
    class_id = box.cls[0].item()
    prob = round(box.conf[0].item(), 2)
    output.append([ x1, y1, x2, y2, result.names[class_id], prob ]) #prints match to output
    
    #modify the image to show the detected plate and text
    cv.rectangle(image, (x1,y1), (x2, y2), (255,0,0), 8)
    
    #crop the detected plate and greyscale it for text detection
    plateImage = image[y1:y2, x1:x2] 
    license_plate_crop_gray = cv.cvtColor(plateImage, cv.COLOR_BGR2GRAY)

    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_gray, image)
    cv.putText(image, str(license_plate_text), (int((int(x1) + int(x2)) / 2) - 70, int(y1) - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 6)
    cv.putText(image, str(license_plate_text), (int((int(x1) + int(x2)) / 2) - 70, int(y1) - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)


#shows output image
plt.imshow(image, cmap="gray")
plt.axis('off')
plt.title("plate image")
plt.show()

    
    






    