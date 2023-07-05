# ML_vww
A tool of Visual Wake Words training and deployment with tflite.
- What is Visual Wake Words: [VWW](https://paperswithcode.com/paper/visual-wake-words-dataset)
## 1. First step
### 1. Install virtual env  
- If you havn't install [NuEdgeWise](https://github.com/OpenNuvoton/NuEdgeWise), please follow the steps to install python virtual env and ***choose `NuEdgeWise_env`***.
- Skip if you have done.
### 2. Running
- The `vww.ipynb` will help you prepare data, train the model, and finally convert to tflite & c++ file.

## 2. Work Flow
### 1. data prepare
- VWW is using COCO dataset, so first way is using our [ML_tf2_object_detection_nu](https://github.com/OpenNuvoton/ML_tf2_object_detection_nu) tool to download. The second way is go to its [website](https://cocodataset.org/#home). 
- Please choose `Train` tab & `COCO Dataset Prepare` accordion in `vww.ipynb` to prepare the vww. For example, person is the VWW's object for classification. Of course you can choose other object/label in COCO dataset for your custom training, but maybe the training data isn't balance enough to get a good result.
- If you have custom dataset of with object and without object label, please use our [ML_tf2_image_classfication_nu](https://github.com/OpenNuvoton/ML_tf2_image_classfication_nu). Because VWW is a two kind image classification with a small enough model to fit in edge device, we can use image_classfication tool to train with small size model, for example mobileNetv2 96X96X3 with alpha=0.35.

### 2. training
- `vww.ipynb` offers some attributes for training configuration. This tool offers mobileNetV1,V2&V3 which reference from [keras](https://keras.io/api/applications/mobilenet/)
- The strategy of this VWW (image classification) training is [transfer learning & fine-tunning](https://www.tensorflow.org/tutorials/images/transfer_learning)
- The output is tflite model.

### 3. Test
- Use `vww.ipynb` to test the tflite model.

### 4. Deployment
- Use `Deployment` tab in `vww.ipynb` to convert tflite model to C source/header files.


## 3. Inference code
- MCU(tflite):


