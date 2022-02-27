# FDSiamFC
    In recent years, Siamese-based trackers have shown remarkable improvement in visual tracking. The general trends are interested in making deeper and more complicated networks to pursue higher accuracy. However, these advances result in cumbersome trackers with respect to size and speed, which hinder the deployment of deep trackers in edge devices. This paper formulates the convolutional layer as an ensemble learner and finds redundancy problems in convolutional neural networks (CNN) coming from numerous useless and duplicate convolutional kernels. To solve the redundancy problem, we propose an hourglass-like Feature Distilling (FD) module consisting of two pointwise convolutional layers. It learns the coupling relationship between convolutional kernels on the Video tracking dataset end-to-end. With the learned coupling relationship, we successfully decoupled the convolutional kernels and compressed CNN in width. Compared with the existing deep compression methods, our FD module is more concise and introduces no additional structure. Extensive experiments on OTB50, TOB100, UAV123, UAV20L, TColor128 demonstrate that our proposed method compresses the fully convolutional Siamese Networks (SiamFC) to get FDSiamFC with an average performance loss of only 1% while achieving 42.6× parameters compression, 21.4× fewer float point computation, 6.8× fewer RAM overhead, and 2.2× speed up.

## Tracking performance on multiple benchmark
| Benchamrk | Sq | OTB50 |  | OTB100 |  | UAV20L |  | UAV123 |  | TColor128 | | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Tracker | \-\- | AUC | DP | AUC | DP | AUC | DP | AUC | DP | AUC | DP | \_\_ |
| SiamFC | 1 | 0.516 | 0.692 | 0.583 | 0.771 | 0.384 | 0.579 | 0.478 | 0.697 | 0.506 | 0.707 | 141.5 |
| FDSiamFC_30 | 0.3 | 0.542 | 0.753 | 0.595 | 0.802 | 0.399 | 0.544 | 0.497 | 0.69 | 0.523 | 0.722 | 306.5 |
| FDSiamFC_15 | 0.15 | 0.53 | 0.724 | 0.58 | 0.786 | 0.404 | 0.566 | 0.492 | 0.693 | 0.494 | 0.683 | 319.9 |

Tracking result on each benchmark is available at  https://pan.baidu.com/s/1bItnRu4Eg7D8U1I5UHWr7A  key: jlii  Include 'txt' and 'mat' format, suitable for got10k toolbox and OTB toolbox respectively.

## Compression results
| Tracker | SiamFC | FDSiamFC_30 | FDSiamFC_15 |
| :--- | :---: | :---: | :---: |
| Model size(MB) | 8.609 | 0.821 | 0.217 |
| Computional cost GFLOPs | 5.453 | 0.699 | 0.255 |
| FPS | 141.5 | 306.5 | 319.9 |

## Guide for FDSiamFC'S Training and Tracking.




