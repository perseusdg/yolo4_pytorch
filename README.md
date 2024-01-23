
# Pytorch-YOLOv4

![Python Version](https://img.shields.io/static/v1?label=python&message=3.6|3.8&color=green)
![PyTorch Version](https://img.shields.io/static/v1?label=pytorch&message=1.4|2.0&color=green)
[![License: Apache 2.0](https://img.shields.io/static/v1?label=license&message=Apache2&color=green)](./License.txt)

A minimal PyTorch implementation of YOLOv4, based on the paper [YOLOv4](https://arxiv.org/abs/2004.10934) and the source code provided by [Hank ai Darknet](https://github.com/hank-ai/darknet). 

Features:
- Inference
- Training (including Mosaic augmentation)

## Repository Structure

```plaintext
├── README.md
├── cfg
├── cfg.py                  - Configuration for training
├── data
├── dataset.py              - Dataset utilities
├── demo.py                 - Demo script for PyTorch model
├── demo_darknet2onnx.py    - Convert Darknet model to ONNX
├── demo_pytorch2onnx.py    - Convert PyTorch model to ONNX
├── models.py               - PyTorch model definitions
├── tool
│   ├── camera.py             - Camera demo script
│   ├── coco_annotation.py    - COCO dataset generator
│   ├── config.py
│   ├── darknet2pytorch.py    - Convert Darknet model to PyTorch
│   ├── region_loss.py
│   ├── utils.py
│   └── yolo_layer.py
├── train.py               - Script for training models
└── weight                 - Weights directory (for Darknet models)
```

Weights Download
----------------

### Darknet Weights

*   Baidu: [Download](https://pan.baidu.com/s/1dAGEW8cm-dqK14TbhhVetA) (Extraction code: dm5b)
*   Google Drive: [Download](https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT)

### PyTorch Converted Weights

You can convert weights using `darknet2pytorch`, or download pre-converted models:

*   Baidu:
    *   yolov4.pth: [Download](https://pan.baidu.com/s/1ZroDvoGScDgtE1ja_QqJVw) (Extraction code: xrq9)
    *   yolov4.conv.137.pth: [Download](https://pan.baidu.com/s/1ovBie4YyVQQoUrC3AY0joA) (Extraction code: kcel)
*   Google Drive:
    *   yolov4.pth: [Download](https://drive.google.com/open?id=1wv_LiFeCRYwtpkqREPeI13-gPELBDwuJ)
    *   yolov4.conv.137.pth: [Download](https://drive.google.com/open?id=1fcbR0bWzYfIEdLJPzOsn4R5mlvR6IQyA)

Training with YOLOv4
--------------------

[Guide to train YOLOv4 on your data](Use_yolov4_to_train_your_own_data.md)

1.  Download weights as per the above links.
2.  Prepare your dataset. For COCO dataset, use `tool/coco_annotation.py`.
3.  Train the model by setting parameters in `cfg.py`:
    
    shCopy code
    
    `python train.py -g [GPU_ID] -dir [Dataset directory] ...`
    

Inference
---------

### Performance on MS COCO dataset

Using pretrained Darknet weights from [https://github.com/hank-ai/darknet).

#### val2017 dataset (input size: 416x416)

| Model type | AP | AP50 | AP75 | APS | APM | APL |
| --- | --- | --- | --- | --- | --- | --- |
| DarkNet (YOLOv4 paper) | 0.471 | 0.710 | 0.510 | 0.278 | 0.525 | 0.636 |


#### testdev2017 dataset (input size: 416x416)

| Model type | AP | AP50 | AP75 | APS | APM | APL |
| --- | --- | --- | --- | --- | --- | --- |
| DarkNet (YOLOv4 paper) | 0.412 | 0.628 | 0.443 | 0.204 | 0.444 | 0.560 |


### Different Inference Options

*   Directly use Darknet model and weights:    
    ```
    python demo.py -cfgfile <cfgFile> -weightfile <weightFile> -imgfile     <imgFile>```    
    
*   Use PyTorch weights (`.pth` file): 
    ```
    python models.py <num_classes> <weightfile> <imgfile> <IN_IMAGE_H>     <IN_IMAGE_W> <namefile(optional)>```
    
*   Convert to ONNX and use ONNX for inference (see sections 3 and 4).
*   Convert to TensorRT engine and use for inference (see section 5).

### Inference Output

Two outputs from inference:

1.  Bounding box locations: `[batch, num_boxes, 1, 4]` (x1, y1, x2, y2).
2.  Scores for each class: `[batch, num_boxes, num_classes]`.

Currently, a small post-processing including NMS (Non-Maximum Suppression) is required. Efforts are ongoing to minimize post-processing time.

Installation Guide
------------------

### Requirements

Ensure these prerequisites are installed:

*   **Anaconda**: [Download here](https://www.anaconda.com/download/)
*   **CUDA Toolkit**: [Download here](https://developer.nvidia.com/cuda-toolkit)
*   **cuDNN**: [Download here](https://developer.nvidia.com/cudnn)

### Setting Up the Environment

1.  Create and activate a new Conda environment:
    ```
    cd C:\
    conda create -n onnx python=3.8 conda activate onnx  
    git clone https://github.com/lordofkillz/yolo4_pytorch.git 
    cd pytorch-YOLOv4
    pip install -r requirements.txt
    ```
    
3.  Install PyTorch compatible with your CUDA version: [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/#linux-and-windows-6).

# 3. Darknet2ONNX

- **This script is to convert the official pretrained darknet model into ONNX**

- **Run python script to generate ONNX model and run the demo**
 **To display the image with detections**
```
python demo_darknet2onnx.py cfg/yolov4-tiny.cfg data/coco.names yolov4-tiny.weights data/dog.jpg 1
```
  
**To save image with prediction**
```
python demo_darknet2onnx.py cfg/yolov4-tiny.cfg data/coco.names yolov4-tiny.weights data/dog.jpg 1 predictions_onnx.jpg
```

    
## 3.1 Dynamic or static batch size

- **Positive batch size will generate ONNX model of static batch size, otherwise, batch size will be dynamic**
    - Dynamic batch size will generate only one ONNX model
    - Static batch size will generate 2 ONNX models, one is for running the demo (batch_size=1)

# 4. Pytorch2ONNX

- **You can convert your trained pytorch model into ONNX using this script**

 **Run python script to generate ONNX model and run the demo**
 
    python demo_pytorch2onnx.py <weight_file> <image_path> <batch_size> <n_classes> <IN_IMAGE_H> <IN_IMAGE_W>
    
For example:

    python demo_pytorch2onnx.py yolov4.pth dog.jpg 8 80 416 416
    
## 4.1 Dynamic or static batch size

- **Positive batch size will generate ONNX model of static batch size, otherwise, batch size will be dynamic**
    - Dynamic batch size will generate only one ONNX model
    - Static batch size will generate 2 ONNX models, one is for running the demo (batch_size=1)

**ONNX to TensorRT Conversion**

Follow these steps to convert ONNX models to TensorRT.
 A. Download TensorRT

**Download the TensorRT zip file from NVIDIA**:
[Download here](https://developer.nvidia.com/tensorrt)

 B. Extract the Zip File to `C:\`. 
 
 C. Set Up the TensorRT Environment.

3.  **Open an Anaconda or Python prompt** and change the directory to the extracted TensorRT folder: #shorten the TensorRT folder name to reflect.**
open anacodna prompt.
```
cd:\tensorrt-8.6.1
conda activate onnx
```
4.Install the required TensorRT wheels using pip. Make sure to install them in the following order:

```
  pip install python\tensorrt-8.6.1-cp38-none-win_amd64.whl
  pip install graphsurgeon\graphsurgeon-0.4.6-py2.py3-none-any.whl
  pip install uff\uff-0.6.9-py2.py3-none-any.whl
  pip install onnx_graphsurgeon\onnx_graphsurgeon-0.3.12-py2.py3-none-any.whl
  ```

D. `copy the files from tensorrt-8.6.1.6\lib folder to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin`.

E. `Cd C:\tensorrt-8.6.1.6\bin`

- **Run the following command to convert YOLOv4 ONNX model into TensorRT engine**

    ```
    trtexec --onnx=yolov4_1_3_416_416_static.onnx --saveEngine=yolov4tiny.engine --workspace=2048 --fp16 --explicitBatch
    ```
    
- Note: If you want to use int8 mode in conversion, extra int8 calibration is needed.

## 5.2 Convert from ONNX of dynamic Batch size

**Run the following command to convert YOLOv4 ONNX model into TensorRT engine**

    
    trtexec --onnx=<onnx_file> \ --minShapes=input:<shape_of_min_batch> --optShapes=input:<shape_of_opt_batch> --maxShapes=input:<shape_of_max_batch> \ --workspace=<size_in_megabytes> --saveEngine=<engine_file> --fp16
    
- For example:
  `
  trtexec --onnx=yolov4_-1_3_320_512_dynamic.onnx \--minShapes=input:1x3x320x512 --optShapes=input:4x3x320x512 --maxShapes=input:8x3x320x512 \ --workspace=2048 --saveEngine=yolov4_-1_3_320_512_dynamic.engine --fp16
`
    

## 5.3 Run the demo
**To display the image with detections**
```
python demo_trt.py yolov4tiny.engine data/dog.jpg 416 416
```

**To save the image with detections**
```
python demo_trt.py yolov4tiny.engine data/dog.jpg 416 416 predictions_trt.jpg
```

- This demo here only works when batchSize is dynamic (1 should be within dynamic range) or batchSize=1, but you can update this demo a little for other dynamic or static batch sizes.
    
- Note1: input_H and input_W should agree with the input size in the original ONNX file.
    
- Note2: extra NMS operations are needed for the tensorRT output. This demo uses python NMS code from `tool/utils.py`.


# 6. ONNX2Tensorflow

- **First:Conversion to ONNX**

    tensorflow >=2.0
    
    1: Thanks:github:https://github.com/onnx/onnx-tensorflow
    
    2: Run git clone https://github.com/onnx/onnx-tensorflow.git && cd onnx-tensorflow
    Run pip install -e .
    
    Note:Errors will occur when using "pip install onnx-tf", at least for me,it is recommended to use source code installation

# 7. ONNX2TensorRT and DeepStream Inference
  
  1. Compile the DeepStream Nvinfer Plugin 
  
    cd DeepStream
    make 
  
  2. Build a TRT Engine.
  
   For single batch, 

    trtexec --onnx=<onnx_file> --explicitBatch --saveEngine=    <tensorRT_engine_file> --workspace=<size_in_megabytes> --fp16
  
   For multi-batch, 
  
    trtexec --onnx=<onnx_file> --explicitBatch --shapes=input:Xx3xHxW -- optShapes=input:Xx3xHxW --maxShapes=input:Xx3xHxW --        minShape=input:1x3xHxW --saveEngine=<tensorRT_engine_file> --fp16

  Note :The maxShapes could not be larger than model original shape.
  
  3. Write the deepstream config file for the TRT Engine.
   
Reference:
- https://github.com/eriklindernoren/PyTorch-YOLOv3
- https://github.com/marvis/pytorch-caffe-darknet-convert
- https://github.com/marvis/pytorch-yolo3

```
@article{yolov4,
  title={YOLOv4: YOLOv4: Optimal Speed and Accuracy of Object Detection},
  author={Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao},
  journal = {arXiv},
  year={2020}
}
```
