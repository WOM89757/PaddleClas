## Qualitycheck Application

`resnet50_vd` version info by qualitycheck datasets

| model       | version | epoch | label_smoothing | dataSet       | inference dir                        | onnx dir                        |
| ----------- | ------- | ----- | --------------- | ------------- | ------------------------------------ | ------------------------------- |
| resnet50_vd | v1.0    | e30   | 0.1             | img5.1(0.5)   | inference_qc/30-img5.1               | models/resnet50_vd_qc_1.0.onnx  |
| resnet50_vd | v1.0.1  | e30   | 0.1             | img5.1.1(0.8) | inference_qc/30-img5.1.1-epsilon-0.1 |                                 |
| resnet50_vd | v1.0.1  | e30   | 0               | img5.1.1(0.8) | inference_qc/30-img5.1.1             | models/resnet50_vd_qc_1.0.1onnx |
| resnet50_vd | v1.1  | e30   | 0.1               | img6.0(0.7) | inference_qc/30-img6.0             | models/resnet50_vd_qc_6.0.onnx |
| resnet50_vd | v1.2  | e30   | 0.1               | img6.1.1(0.7) | inference_qc/30-img6.1.1             | models/resnet50_vd_qc_1.2.onnx |

### label smoothing

![label smoothing compared](./img/20220520-175029.png)
#### Confusion Matrix
![label smoothing compared](./img/Confusion_Matrix—inference_qc-30-img5.1.1-no-label-smoothing.png)
![label smoothing compared](./img/Confusion_Matrix—inference_qc-30-img5.1.1-label-smoothing.png)
#### Roc Curve
![label smoothing compared](./img/Roc_Curve—inference_qc-30-img5.1.1-no-label-smoothing.png)
![label smoothing compared](./img/Roc_Curve—inference_qc-30-img5.1.1-label-smoothing.png)

### Model export and convert

```bash
# train
python tools/train.py -c ./ppcls/configs/qualitycheck/ResNet/ResNet50_vd_qc.yaml -o Arch.pretrained=True

# infer
python tools/infer.py --config ./ppcls/configs/qualitycheck/ResNet/ResNet50_vd_qc.yaml  --override Global.pretrained_model=./output_qc/ResNet50_vd/best_model --override Infer.infer_imgs=dataset/img4.0/test/chaodi01/801_00007_image_3.jpg

# export
python tools/export_model.py --config ./ppcls/configs/qualitycheck/ResNet/ResNet50_vd_qc.yaml --override Global.pretrained_model=./output_qc/ResNet50_vd/best_model

# convert
paddle2onnx --model_dir=inference_qc/30-img6.0/  --model_filename=inference.pdmodel --params_filename=inference.pdiparams --save_file=models/resnet50_vd_qc_1.1.onnx --opset_version=10  --enable_onnx_checker=True
```
