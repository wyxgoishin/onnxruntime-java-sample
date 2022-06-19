This project shows a sample on Optical Flow Inference with onnxruntime-java.
Model file are excluded due to confidential.
If you wish to test it yourself, you can download RAFT model from
[official repository](https://github.com/princeton-vl/RAFT) and export
pretrained model to onnx format. It's also to be noted that I have
made specific changes as list below:
1. run project:
  - To run the project, at least jdk 9 is required
  - Because this project depends on opencv 4.5.5 and I have installed it manually for packaging,
you need to install it through maven manually as well. The code is as below:
```cmd
mvn install:install-file -DgroupId=org -DartifactId=opencv -Dversion="4.5.5" -Dpackaging="jar" -Dfile=<path-to-jar-file>
```
  - If you change the version of opencv to be used, don't forget to change the 'OpencvDllName' field in 
MiscUtil.java as well
2. model input: 
  - image1 and image2 (both H * W * C tensor (float32), and its' height and width
will be padded with 'sintel' fashion before inference to be divisible by 8)
  - iters are fixed at 20 (as currently I found no method to pass a
scalar to onnx model. Passing a tensor with shape as iters doesn't work
fine)
  - up-sampling is implemented by `upflow8()` rather than `upsample_flow()`, as onnx don't support 
`torch.softmax()` which is used in `upsample_flow()`
  - `torch.nn.function.grid_sample()` is replaced by `mmcv.op.point_sample.bilinear_grid_sample()`
as previous one is not supported for exporting runnable model
3. model output:
  - flow (B * H * W * C tensor (float32), which is the uppadded result of up-sampled 
flow)
4. save flow:
  - Currently, I don't found a proper way to save image with depth as CV_32F.
I have tested tiff and hdr, but both can't store the origin value. May you can try
exr format, but that needs extra compiling of opencv.
  - So to solve that problem, I currently save flow in kitti-format. That is to say, 
kitti-flow = (flow * 64 + 2 ** 15).astype(uint16) and RGB channels store 1, 
kitti-flow-v, kitti-flow-u separately. You can use code below in python to read:
```python
  import cv2
  import numpy as np
  kitti_flow_path = 'xxx'
  kitti_flow = cv2.imread(kitti_flow_path, -1)
  flow = (kitti_flow[:, :, ::-1][:, :, :2].astype(np.float32) - 32768) / 64 
```
