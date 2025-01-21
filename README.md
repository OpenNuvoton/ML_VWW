# ML_vww
This tool focuses on the training and deployment of Visual Wake Words using TFLite.
- What is Visual Wake Words: [VWW](https://paperswithcode.com/paper/visual-wake-words-dataset)
## 1. First step
### 1. Install virtual env  
- If you haven't installed [NuEdgeWise](https://github.com/OpenNuvoton/NuEdgeWise), please follow these steps to install Python virtual environment and ***choose `NuEdgeWise_env`***.
- Skip if you have done.
### 2. Running
- The `vww.ipynb` will help you prepare data, train the model, and finally convert it to a TFLite and C++ file.

## 2. Work Flow
### 1. data prepare
- VWW utilizes the COCO dataset, and there are two ways to obtain it. The first method is to use our [ML_Object_Detection](https://github.com/OpenNuvoton/ML_Object_Detection) tool for downloading. Alternatively, you can visit the COCO dataset [website](https://cocodataset.org/#home).
- Please select the Train tab and expand the COCO Dataset Prepare accordion in vww.ipynb to prepare the Visual Wake Words (VWW) dataset. For instance, if you intend to classify the object 'person' using VWW, you can choose it as your target object. However, keep in mind that selecting other objects/labels from the COCO dataset for custom training may result in an imbalance of training data, potentially affecting the quality of the results.
- If you have a custom dataset with object and non-object labels, please utilize our [ML_Image_Classification](https://github.com/OpenNuvoton/ML_Image_Classification) tool. Since VWW involves two-class image classification and requires a compact model deployable on edge devices, you can use the image_classification tool to train a lightweight model such as MobileNetV2 with dimensions of 96x96x3 and an alpha value of 0.35.

### 2. training
- vww.ipynb provides various attributes for training configuration. This tool offers MobileNetV1, V2, and V3 models, which are referenced from [keras](https://keras.io/api/applications/mobilenet/).
- The training strategy for this Visual Wake Words (VWW) is based on [transfer learning & fine-tunning](https://www.tensorflow.org/tutorials/images/transfer_learning).
- The output is tflite model.

### 3. Test
- Use `vww.ipynb` to test the tflite model.

### 4. Deployment
- Utilize the `Deployment` tab in `vww.ipynb` to convert the TFLite model to C source/header files.


## 3. Inference code 
- The ML_SampleCode repositories are private. Please contact Nuvoton to request access to these sample codes. [Link](https://www.nuvoton.com/ai/contact-us/)
  - [ML_M460_SampleCode (private repo)](https://github.com/OpenNuvoton/ML_M460_SampleCode)
      - `tflu_vww`: real time inference code
- [M55M1BSP](https://github.com/OpenNuvoton/M55M1BSP/tree/master/SampleCode/MachineLearning)
  - `VisualWakeWords`: real time inference code
  


