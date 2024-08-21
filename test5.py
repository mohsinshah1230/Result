from roboflow import Roboflow
import supervision as sv
import cv2

# Initialize Roboflow model
rf = Roboflow(api_key="1gcPa1rF1UMHGBEVWtfp")
project = rf.workspace().project("supermarket-empty-shelf-detector")
model = project.version(3).model

# Initialize annotators
label_annotator = sv.LabelAnnotator()
bounding_box_annotator = sv.BoxAnnotator()

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to avoid exceeding the Roboflow API size limit
    resized_frame = cv2.resize(frame, (640, 480))

    # Save the resized frame to a temporary file
    temp_image_path = "temp_frame.jpg"
    cv2.imwrite(temp_image_path, resized_frame)

    # Predict using the model
    result = model.predict(temp_image_path, confidence=40, overlap=30).json()

    # Extract labels from the predictions
    labels = [item["class"] for item in result["predictions"]]

    # Convert Roboflow predictions to Supervision detections
    detections = []
    for prediction in result['predictions']:
        xmin = prediction['x'] - prediction['width'] / 2
        ymin = prediction['y'] - prediction['height'] / 2
        xmax = prediction['x'] + prediction['width'] / 2
        ymax = prediction['y'] + prediction['height'] / 2
        detections.append(sv.Detection(box=[xmin, ymin, xmax, ymax], label=prediction['class'], score=prediction['confidence']))

    # Annotate frame
    annotated_frame = bounding_box_annotator.annotate(scene=resized_frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    # Display annotated frame
    cv2.imshow('Live Detection', annotated_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
