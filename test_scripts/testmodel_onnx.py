# import cv2
# import numpy as np
# import onnxruntime
# import pytesseract
# import PyPDF2

# # Load the image
# image = cv2.imread("test-materials/nightmare.jpeg")

# # Check if the image has been loaded successfully
# if image is None:
#     raise ValueError("Failed to load the image")
    
# # Get the shape of the image
# height, width = image.shape[:2]

# # Make sure the height and width are positive
# if height <= 0 or width <= 0:
#     raise ValueError("Invalid image size")



# # Set the desired size of the resized image
# dsize = (640, 640)

# # Resize the image using cv2.resize
# resized_image = cv2.resize(image, dsize)

# # Display the resized image
# # cv2.imshow("Resized Image", resized_image)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# # Load the ONNX model
# session = onnxruntime.InferenceSession("best-roboflow.onnx")

# # Check if the model has been loaded successfully
# if session is None:
#     raise ValueError("Failed to load the model")

# # Get the input names and shapes of the model
# inputs = session.get_inputs()
# for i, input_info in enumerate(inputs):
#     print(f"Input {i}: name = {input_info.name}, shape = {input_info.shape}")

# # Run the ONNX model
# input_name = session.get_inputs()[0].name
# output_name = session.get_outputs()[0].name
# prediction = session.run([output_name], {input_name: image})[0]

# # Postprocess the prediction to obtain the labels
# labels = postprocess(prediction)

# # Use PyTesseract to extract the text from the image
# text = pytesseract.image_to_string(image)

# # Print the labels and the text
# print("Labels:", labels)
# print("Text:", text)


# THIS CODE ACTUALLY WORKS

import cv2
from yolov8 import YOLOv8

# # Initialize the webcam
# video_path = "test-materials/rleaves.MOV"
# cap = cv2.VideoCapture(0) #VideoCapture(0) for webcam, (video_path) for video

# # Initialize YOLOv8 object detector
# model_path = "best-roboflow.onnx"
# yolov8_detector = YOLOv8(model_path, conf_thres=0.75, iou_thres=0.5)

# cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
# while cap.isOpened():

#     # Read frame from the video
#     ret, frame = cap.read()

#     if not ret:
#         break

#     # Update object localizer
#     boxes, scores, class_ids = yolov8_detector(frame)

#     combined_img = yolov8_detector.draw_detections(frame)
#     cv2.imshow("Detected Objects", combined_img)

#     # Press key q to stop
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break




# Initialize YOLOv8 object detector
model_path = "best-roboflow.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.75, iou_thres=0.5)

imgnumber=9089

for i in range(34):
    frame = cv2.imread("test-materials/cage_images/IMG_{}.HEIC".format(imgnumber))
    # frame = "test-materials/cage_images/{}.HEIC".format(imgnumber)

    boxes, scores, class_ids = yolov8_detector(frame)

    combined_img = yolov8_detector.draw_detections(frame)
    cv2.imshow("Detected Objects", combined_img)
    imgnumber += 1