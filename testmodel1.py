# run in terminal

# yolov8 on uploaded images/videos
yolo task=detect \
mode=predict \
model= best-roboflow.pt \
conf=0.6 \
source='test-materials/cage_images/IMG_9186_60.jpg' \
show = True

# yolov8 on webcam (from https://towardsdatascience.com/enhanced-object-detection-how-to-effectively-implement-yolov8-afd1bf6132ae)
yolo detect predict model=best-roboflow.pt source=0 show=True