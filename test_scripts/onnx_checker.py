# import onnx
# onnx_model = onnx.load('best-roboflow.onnx')

# # Check the model
# try:
#     onnx.checker.check_model(onnx_model)
# except onnx.checker.ValidationError as e:
#     print('The model is invalid: %s' % e)
# else:
#     print('The model is valid!')

###

import numpy as np
import onnxruntime as rt

image = 'test-materials/fruits.jpg'
sample = np.expand_dims(image, axis=0)

onnx_path = 'best-roboflow.onnx'
sess = rt.InferenceSession(onnx_path)
x_name = sess.get_inputs()[0].name
y1_name = sess.get_outputs()[0].name
y2_name = sess.get_outputs()[1].name
y3_name = sess.get_outputs()[2].name
outPred = sess.run([y1_name, y2_name, y3_name], {x_name: sample})

print(outPred)