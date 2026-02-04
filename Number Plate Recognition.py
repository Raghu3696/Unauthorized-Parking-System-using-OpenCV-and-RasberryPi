import cv2
import os
import matplotlib.pyplot as plt
from easyocr import Reader


# CONFIGURATION


# Folder containing car images
IMAGE_FOLDER = ""

# Initialize EasyOCR reader
# Use ['en'] for English plates, ['ar'] for Arabic plates
reader = Reader(['ar'], gpu=False, verbose=False)


# PROCESS EACH IMAGE


for filename in os.listdir(IMAGE_FOLDER):

    if not filename.lower().endswith(".jpg"):
        continue

    image_path = os.path.join(IMAGE_FOLDER, filename)

    # Read and resize image
    car_image = cv2.imread(image_path)
    car_image = cv2.resize(car_image, (800, 600))

    
    # IMAGE PREPROCESSING
    

    # Convert to grayscale
    gray = cv2.cvtColor(car_image, cv2.COLOR_BGR2GRAY)

    # Reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blur, 10, 200)

    
    # LICENSE PLATE DETECTION
    

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area and keep largest ones
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    plate_contour = None

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        # License plate is usually rectangular
        if len(approx) == 4:
            plate_contour = approx
            break

    
    # OCR (TEXT RECOGNITION)
    

    if plate_contour is not None:

        x, y, w, h = cv2.boundingRect(plate_contour)

        # Ensure ROI stays inside image
        x, y = max(0, x), max(0, y)
        plate_roi = gray[y:y+h, x:x+w]

        # Read text using OCR
        detections = reader.readtext(plate_roi)

        if len(detections) == 0:
            cv2.putText(
                car_image,
                "Unable to read license plate",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )
        else:
            text = detections[0][1]
            confidence = detections[0][2] * 100

            # Draw plate contour
            cv2.drawContours(car_image, [plate_contour], -1, (255, 0, 0), 3)

            # Display detected text
            label = f"{text} ({confidence:.2f}%)"
            cv2.putText(
                car_image,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            print(f"Detected Plate: {label}")

    
    # DISPLAY RESULTS
    

    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(car_image, cv2.COLOR_BGR2RGB))
    plt.title("License Plate Detection & OCR")
    plt.axis("off")
    plt.show()
