# FDSiamFC.

In recent years, Siamese-based trackers have shown remarkable improvement in visual tracking. The general trends are interested in making deeper and more complicated networks to pursue higher accuracy. However, these advances result in cumbersome trackers with respect to size and speed, which hinder the deployment of deep trackers in edge devices. This paper formulates the convolutional layer as an ensemble learner and finds redundancy problems in convolutional neural networks (CNN) coming from numerous useless and duplicate convolutional kernels. To solve the redundancy problem, we propose an hourglass-like Feature Distilling (FD) module consisting of two pointwise convolutional layers. It learns the coupling relationship between convolutional kernels on the Video tracking dataset end-to-end. With the learned coupling relationship, we successfully decoupled the convolutional kernels and compressed CNN in width. Compared with the existing deep compression methods, our FD module is more concise and introduces no additional structure. Extensive experiments on OTB50, TOB100, UAV123, UAV20L, TColor128 demonstrate that our proposed method compresses the fully convolutional Siamese Networks (SiamFC) to get FDSiamFC with an average performance loss of only 1% while achieving 42.6× parameters compression, 21.4× fewer float point computation, 6.8× fewer RAM overhead, and 2.2× speed up.

## Tracking performance on multiple benchmark.
| Benchamrk | Sq | OTB50 |  | OTB100 |  | UAV20L |  | UAV123 |  | TColor128 | | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Tracker | \-\- | AUC | DP | AUC | DP | AUC | DP | AUC | DP | AUC | DP | \_\_ |
| SiamFC | 1 | 0.516 | 0.692 | 0.583 | 0.771 | 0.384 | 0.579 | 0.478 | 0.697 | 0.506 | 0.707 | 141.5 |
| FDSiamFC_30 | 0.3 | 0.542 | 0.753 | 0.595 | 0.802 | 0.399 | 0.544 | 0.497 | 0.69 | 0.523 | 0.722 | 306.5 |
| FDSiamFC_15 | 0.15 | 0.53 | 0.724 | 0.58 | 0.786 | 0.404 | 0.566 | 0.492 | 0.693 | 0.494 | 0.683 | 319.9 |

Tracking result on each benchmark is available at  https://pan.baidu.com/s/1bItnRu4Eg7D8U1I5UHWr7A  key: jlii.  Include 'txt' and 'mat' format, suitable for got10k toolbox and OTB toolbox respectively.

## Compression results.
| Tracker | SiamFC | FDSiamFC_30 | FDSiamFC_15 |
| :--- | :---: | :---: | :---: |
| Model size(MB) | 8.609 | 0.821 | 0.217 |
| Computional cost GFLOPs | 5.453 | 0.699 | 0.255 |
| FPS | 141.5 | 306.5 | 319.9 |

## Guide for FDSiamFC'S Training and Tracking.
### Compress SiamFC to get FDSiamFC
#### Create virtual enviroment
The packages neeeded for FDSiamFC is avilable at requirement.txt. For RTX30 or laster GPU you may need to use pytorch1.7 or laster. I recomand you to use Pytorch1.9 instead of 1.7 since 1.7 does not well compatible with Group convolutional and big kernel convolutional(like 5×5 convolutional kernel).
#### Preprocess on the training dataset.
You may take ILSVRC15, GOT10K, TrackingNet as the training dataset of FDSiamFC (In our work we only use GOT10K as the training dataset)
go to './tools/crop_train_dataset_got10k.py'
modify the 'got10k_dir' and 'cropped_data_dir' in line 111 and line 112.
got10k_dir is the folder where you store the GOT10K dataset.
cropped_data_dir is the folder you want to store the cropped training datasets.
you may need 30 GB free space to store the cropped training datasets. NVME SSD is stronglly recomand, it may faster you training process more than 10×.
Run
  crop_train_dataset_got10k.py
It will take 2 hours to crope the whole training dataset.
After cropped the training dataset, you may need a meta data file. you can generate it by youself fallow the codes in './siamfc/datasets.py' line 25-46 or download it from :https://pan.baidu.com/s/1BDiYbmL-iSOQ866yCRHR8A  key 20td. 

#### Training FDSiamFC.
The FDSiamFC is compressing the FDSiamFC layer by layer throught FD module. On each layer we will test the compressed model on OTB2013 dataset and chose a suitable model for the next compressing step. So, you may neet to download the OTB2013 dataset and set the folder path in './tools/fdModel_evaluate.py' line 21. You may use other benchmark to evaluate the compressed model insted.
Go to './tools/train_FDSiamFC.py'
modify the 'root_dir' in line 16.
root_dir is the path where you stored the cropped training dataset.
run
  train_FDSiamFC.py
It may take 20 hours to compress the whole SiamFC. During the compressing process, the compress result will printed in './tools/train_log.txt'.

#### Transform the compressed model.
After the traning process, the FD model's encode layer successfully learn the couplling relatinship between the convolutional kernels in each layer. Now we need to used these relationship to decoupling the convolutional kernels.
Go to './tools/model_transform_high_precision.py' (note that: './tools/The model_transform.py' achieves same function but possibly loss precision)
modify the 'model_path' in line 54. (the best compressed model file path is avilable at the train_log.txt file you can directly copy the path on the train log)
run
  model_transform_hight_precision.py
The decoupled model will be avilable at './tool/transformed_model.pth'. Now the train process is over and we have successfully compress the SiamFC into FDSiamFC.









