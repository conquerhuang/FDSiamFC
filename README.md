# FDSiamFC.

In recent years, Siamese-based trackers have shown remarkable improvement in visual tracking. The general trends are interested in making deeper and more complicated networks to pursue higher accuracy. However, these advances result in cumbersome trackers with respect to size and speed, which hinders the deployment of deep trackers in edge devices. Due to the tracking scenario complexity and temporal coherence, the backbone network of deep trackers emphasizes the target appearance information more. However, the appearance feature is sensitive to parameter variation, making the traditional deep model compression methods hard to compress a deep tracker. To bridge the gap between deep Siamese trackers and practical use, we propose a new feature distillation algorithm suitable for deep trackers in this paper. Firstly, motivated by the concept of divide-and-conquer, we formulate the feature distillation into a stepwise distillation problem and perform distillation on each minimum unit to relieve the hard-to-distill problem of appearance feature. Secondly, we reconstruct the student model into a combination of convolution kernels and a point-wise convolutional layer, which enables the student model to inherit all the parameters of the teacher model during initialization. Finally, we propose a 3-step warm-up training strategy to address the student model's degradation and structural adaptation problem during training. Extensive experiments on seven benchmarks demonstrate that our proposed method compresses the fully convolutional Siamese Networks SiamFC and its variant Siamd and achieves leading tracking performance with only 2.1 MB model size while running at 225 fps.

## Tracking performance on multiple benchmark.
| Benchamrk | Sq | OTB50 |  | OTB100 |  | UAV20L |  | UAV123 |  | TColor128 | | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Tracker | \-\- | AUC | DP | AUC | DP | AUC | DP | AUC | DP | AUC | DP | \_\_ |
| SiamFC | 1 | 0.516 | 0.692 | 0.583 | 0.771 | 0.384 | 0.579 | 0.478 | 0.697 | 0.506 | 0.707 | 141.5 |
| FDSiamFC_30 | 0.3 | 0.542 | 0.753 | 0.595 | 0.802 | 0.434 | 0.604 | 0.509 | 0.715 | 0.523 | 0.722 | 306.5 |
| FDSiamFC_15 | 0.15 | 0.53 | 0.724 | 0.58 | 0.786 | 0.422 | 0.599 | 0.496 | 0.704 | 0.494 | 0.683 | 319.9 |


## Compression results.
the row results are available at: https://pan.baidu.com/s/1Rwy8iATpv2SfH138QUk0Rw?pwd=1hio key: 1hio 
| Tracker | SiamFC | FDSiamFC_30 | FDSiamFC_15 |
| :--- | :---: | :---: | :---: |
| Model size(MB) | 8.609 | 0.821 | 0.217 |
| Computional cost GFLOPs | 5.453 | 0.699 | 0.255 |
| FPS | 141.5 | 306.5 | 319.9 |

## Guide for FDSiamFC'S Tracking.
### Tracking with FDSiamFC
#### Tracking a single video sequence

Download the pretrained model at: https://pan.baidu.com/s/1xlf5qmlSGGW0IMUqwduT1g?pwd=dq9o  key:dq9o 

Go to './tools/demo_fdsiamfc.py'

Modify the video path you wanna to track. In the demo, we give a video sequence with OTB format. you can change the input parameters in the tracker.track() to track any video sequence you want.

Modify the squeeze_rate parameter in line 20. If you use the trained model by you self, you should set the Sq parameter similar to './tools/train_FDSiamFC.py' in line 19. 

'fdsiamfc_15.pth' → squeeze_rate = [0.15,  0.15, 0.15, 0.15, 0.15]  or

'fdsiamfc_30.pth' → squeeze_rate = [0.3,  0.3, 0.3, 0.3, 0.3]

run

    demo_fdsiamfc.py

Enjoy the tracking process!

### Test on tracking benchmark.
We offer a test file for you to test the trained FDSiamFC on multiple benchmark. You should install the got10k toolbox by 

    pip install got10k

or directly download it from got10k websit and add it to this project.

Go to './tools/test_fdsiamfc.py'

Modify the model_path in line 67 and set the squeeze_rate identical to your model.
Run

    test.py
    
### search for best parameters
We provoide a hyper parameter search code.

Go to './hyper_parameter/esiam_lite_hyper_parameter.py'

Modify the tracker and model, and search the hyper parameters. 

Wait the got10k toolbox. It may take 5-10 minutes to evaluate FDSiamFC. you can choose other Benchmarks to evaluate FDSiamFC by setting the experiments in line 16.

### train and test enviroment

The model and source code is developed and tested on Windows plantform (win 10) CPU i9 9700k GPU RTX3080ti








