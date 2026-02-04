import cv2
import numpy as np
from keras.models import load_model
from easyocr import Reader

# LOAD CNN MODEL (from your existing project)
model = load_model("emptyparkingspotdetectionmodel.h5")

# INIT OCR
reader = Reader(['en'], gpu=False)

# PARKING SLOT COORDINATES (same as your training script)
coordinates = [
    [(20, 8), (58, 88)], [(59, 8), (102, 87)],
    # keep all your coordinates here
]

def detect_empty(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48,48))
    resized = resized.astype("float32") / 255
    resized = np.reshape(resized, (1,48,48,1))
    pred = model.predict(resized)
    return pred[0][0] > 0.5

def read_plate(car_roi):
    gray = cv2.cvtColor(car_roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,10,200)

    contours,_ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for c in contours:
        peri = cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c,0.02*peri,True)
        if len(approx)==4:
            x,y,w,h = cv2.boundingRect(approx)
            plate_roi = gray[y:y+h,x:x+w]
            text = reader.readtext(plate_roi)
            if text:
                return text[0][1]
    return None

cap = cv2.VideoCapture(0)
previous_status = [True]*len(coordinates)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    for i, spot in enumerate(coordinates):
        (x1,y1),(x2,y2) = spot
        roi = frame[y1:y2,x1:x2]

        current_status = detect_empty(roi)

        if previous_status[i] and not current_status:
            plate = read_plate(roi)
            print("Unauthorized Vehicle:", plate)

        previous_status[i] = current_status

        color = (0,255,0) if current_status else (0,0,255)
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

    cv2.imshow("Smart Parking - Raspberry Pi",frame)

    if cv2.waitKey(1)==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()