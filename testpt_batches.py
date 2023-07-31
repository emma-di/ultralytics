from ultralytics import YOLO
model = YOLO('best-roboflow.pt')

imageID = 9186
images = []

while imageID < 9206:
    images.append('/home/reu/Downloads/ultralytics/test-materials/cage_images/IMG_{}.jpg'.format(imageID))
    imageID += 1

model.predict(source = images, save = True, conf = .6)