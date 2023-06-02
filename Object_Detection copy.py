import cv2
import numpy as np

# Load the YOLOv3 model and its configuration
model_config = "yolov3.cfg"
model_weights = "yolov3.weights"
net = cv2.dnn.readNetFromDarknet(model_config, model_weights)

# Get the names of the output layers
layer_names = net.getLayerNames()
#print(layer_names)
output_layers = []
unconnected_layers = net.getUnconnectedOutLayers()
for i in unconnected_layers:
    layer_index = i - 1
    output_layers.append(layer_names[layer_index])


# Load the COCO class labels
with open("coco.names", "r") as f:
    classes = f.read().splitlines()

# Load the input image
image_path = "hotdog.jpg"
image = cv2.imread(image_path)

# Prepare the input image for object detection
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)


# Set the input for the network
net.setInput(blob)

# Run forward pass to perform object detection
outs = net.forward(output_layers)

# Initialize lists to store the detected objects' bounding boxes, confidences, and class labels
boxes = []
confidences = []
class_labels = []

# Process each output layer
for out in outs:
    # Iterate over each detection
    for detection in out:
        # Extract class scores and class label
        scores = detection[5:]
        class_id = np.argmax(scores)
        # print(class_id.shape)
        confidence = scores[class_id]

        # Consider only detections with high confidence
        if confidence > 0.5:
            # Scale the bounding box coordinates back to the original image size
            box = detection[0:4] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
            (centerX, centerY, width, height) = box.astype("int")

            # Calculate the top-left corner coordinates of the bounding box
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))

            # Store the bounding box, confidence, and class label
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            class_labels.append(class_id)

# Apply non-maximum suppression to remove redundant overlapping boxes
indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.3)

# Loop over the remaining detections and draw bounding boxes around detected objects
print(image.shape)
for i in indices:
    label = f"{classes[class_labels[i]]}: {confidences[i]:.2f}"
    #print(classes[class_labels[i]])
    if classes[class_labels[i]] == "hot dog":
        (x, y, w, h) = boxes[i]
        # print(x)
        # print(y)
        ## X y Y son la esquina superior izquierda y el eje Y está invertido 
        # Draw the bounding box and label on the image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the output image
cv2.imshow("Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()