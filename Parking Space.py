# Empty Parking Space

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Initialize the drawing state
drawing = False
ix, iy = -1, -1

# Load the image
img = cv2.imread("C:\\Users\\raghu\\OneDrive\\Desktop\\Unautorized Car Parking\\parkingarea.png")

# Convert the image from BGR to RGB for displaying with matplotlib
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Create a figure and axis for plotting the image
fig, ax = plt.subplots()
im_plot = ax.imshow(img)

# Function to draw a rectangle on the image
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing

    # When the left mouse button is pressed, start drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    # When the mouse is moved, draw a rectangle if drawing is True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_rect = img.copy()
            cv2.rectangle(img_rect, (ix, iy), (x, y), (0, 255, 0), 2)
            im_plot.set_data(img_rect)
            plt.draw()

    # When the left mouse button is released, stop drawing
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
        im_plot.set_data(img)
        plt.draw()
        print("Top Left:", (ix, iy))
        print("Bottom Right:", (x, y))

# Function to handle mouse click events
def onclick(event):
    global ix, iy, drawing

    # If the left mouse button is clicked
    if event.button == 1:
        # If not drawing, start drawing
        if not drawing:
            draw_rectangle(cv2.EVENT_LBUTTONDOWN, int(event.xdata), int(event.ydata), None, None)
        # If drawing, stop drawing
        else:
            draw_rectangle(cv2.EVENT_LBUTTONUP, int(event.xdata), int(event.ydata), None, None)

# Function to handle mouse move events
def onmove(event):
    # If drawing and the mouse coordinates are valid, draw a rectangle
    if drawing and event.xdata is not None and event.ydata is not None:
        draw_rectangle(cv2.EVENT_MOUSEMOVE, int(event.xdata), int(event.ydata), None, None)

# Connect the event handlers to the figure canvas
fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('motion_notify_event', onmove)
fig.canvas.mpl_connect('button_release_event', onclick)

# Display the figure
plt.show()


#Code for Empty Parking space

import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical

# Paths to the training data folders for empty and occupied parking spots
training_data = [
    "C:\\Users\\raghu\\OneDrive\\Desktop\\Unautorized Car Parking\\empty",
    "C:\\Users\\raghu\\OneDrive\\Desktop\\Unautorized Car Parking\\occupied"
]

# Function to load images and their corresponding labels from the given directories
def load_images(training_data):
    images = []
    labels = []
    for i, folder in enumerate(training_data):
        label = i  # 0 for empty, 1 for occupied
        for filename in os.listdir(folder):
            try:
                # Read the image in grayscale
                img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
                # Resize the image to 48x48 pixels
                img = cv2.resize(img, (48, 48))
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Error loading image {os.path.join(folder, filename)}: {e}")
    return np.array(images), np.array(labels)

# Load images and labels
images, labels = load_images(training_data)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Preprocess the images: reshape and normalize
X_train = X_train.reshape(X_train.shape[0], 48, 48, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 48, 48, 1).astype('float32') / 255

# Convert labels to categorical (one-hot encoding)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Create a sequential model
model = Sequential()
# Add convolutional and pooling layers
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# Flatten the output and add dense layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
# Output layer with softmax activation for classification
model.add(Dense(2, activation='softmax'))

# Compile the model with categorical crossentropy loss and adam optimizer
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

# Train the model with training data
model.fit(X_train, y_train, batch_size=64, epochs=50, verbose=1, validation_data=(X_test, y_test))

# Save the trained model to a file
model.save("emptyparkingspotdetectionmodel.h5")
from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Load the pre-trained model
model = load_model("emptyparkingspotdetectionmodel.h5")
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

# List of parking spot coordinates
coordinates = [
    [(20, 8), (58, 88)], [(59, 8), (102, 87)], [(100, 4), (144, 85)], [(145, 8), (184, 87)],
    [(186, 10), (226, 89)], [(225, 9), (268, 90)], [(267, 9), (308, 88)], [(309, 8), (349, 89)],
    [(349, 7), (391, 89)], [(394, 9), (434, 90)], [(436, 10), (474, 91)], [(474, 9), (515, 91)],
    [(517, 12), (573, 94)], [(23, 194), (63, 276)], [(63, 194), (103, 278)], [(103, 197), (142, 277)],
    [(145, 196), (182, 274)], [(187, 197), (227, 278)], [(228, 198), (270, 275)], [(269, 190), (308, 275)],
    [(311, 199), (346, 272)], [(354, 196), (389, 272)], [(396, 196), (433, 273)], [(437, 195), (480, 275)],
    [(487, 201), (511, 273)], [(521, 199), (566, 271)], [(26, 282), (61, 361)], [(65, 284), (103, 359)],
    [(107, 281), (144, 362)], [(152, 287), (175, 365)], [(185, 281), (223, 363)], [(231, 284), (268, 359)],
    [(275, 287), (310, 362)], [(312, 284), (347, 361)], [(353, 284), (389, 363)], [(395, 284), (432, 365)],
    [(437, 285), (470, 364)], [(476, 282), (520, 370)], [(529, 290), (568, 361)]
]

def detect_empty_parking(image, spot):
    x1, y1 = spot[0]
    x2, y2 = spot[1]

    if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
        print("Invalid coordinates for ROI")
        return False

    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        print("Empty ROI")
        return False

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized_roi = cv2.resize(gray_roi, (48, 48))
    resized_roi = resized_roi.astype('float32') / 255
    resized_roi = np.expand_dims(resized_roi, axis=0)
    resized_roi = np.expand_dims(resized_roi, axis=-1)

    prediction = model.predict(resized_roi)
    threshold = 0.01
    if prediction[0][0] > threshold:
        return True
    else:
        return False

# Read the input image
current_image = cv2.imread("C:\\Users\\raghu\\OneDrive\\Desktop\\Unautorized Car Parking\\Parking area 3.png")
empty_count = 0

# Process each parking spot
for spot in coordinates:
    if detect_empty_parking(current_image, spot):
        cv2.rectangle(current_image, spot[0], spot[1], (0, 255, 0), 2)
        empty_count += 1
    else:
        cv2.rectangle(current_image, spot[0], spot[1], (0, 0, 255), 2)

# Add text overlay
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(current_image, f"Empty Spots: {empty_count}", (50, 50), font, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

# Display the image with matplotlib
plt.figure(figsize=(10, 7))
plt.imshow(cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()


# Number Plate Recognition using OCR
import os
from easyocr import Reader
import cv2
import matplotlib.pylab as plt

image_folder = "C:\\Users\\raghu\\OneDrive\\Desktop\\Unautorized Car Parking\\Data"
reader = Reader(['ar'], gpu=False, verbose=False)

# Image upload and processing:

for filename in os.listdir(image_folder):
    if filename.endswith(".jpg"):
        image_path = os.path.join(image_folder, filename)
        car = cv2.imread(image_path)
        car = cv2.resize(car, (800, 600))

        # Image processing for edge detection
        gray = cv2.cvtColor(car, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blur, 10, 200)

        # Identify the panel:
        cont, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cont = sorted(cont, key=cv2.contourArea, reverse=True)[:5]
        plate_cnt = None
        for c in cont:
            arc = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * arc, True)
            if len(approx) == 4:
                plate_cnt = approx
                break

      # Read text from panel using OCR:
        if plate_cnt is not None:
            (x, y, w, h) = cv2.boundingRect(plate_cnt)
            x = max(0, x)
            y = max(0, y)
            plate_roi = gray[y:y + h, x:x + w]

            detection = reader.readtext(plate_roi)

            if len(detection) == 0:
                text = "Impossible to read the text from the license plate"
                cv2.putText(car, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 3)
            else:
                cv2.drawContours(car, [plate_cnt], -1, (255, 0, 0), 3)
                text = f"{detection[0][1]} {detection[0][2] * 100:.2f}%"
                cv2.putText(car, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                print(text)

        # Display the original image
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(car, cv2.COLOR_BGR2RGB))
        plt.title("Image")

        # If the board is found, display a thumbnail of the board
        if plate_cnt is not None:
            plt.subplot(1, 3, 2)
            plt.imshow(cv2.cvtColor(plate_roi, cv2.COLOR_BGR2RGB))
            plt.title("License Plate")

            # Display the original image with the panel selected
            plt.subplot(1, 3, 3)
            plt.imshow(cv2.cvtColor(car, cv2.COLOR_BGR2RGB))
            plt.title("Image with Contours")

        plt.show()

