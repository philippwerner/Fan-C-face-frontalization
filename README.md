# Fan-C Face Frontalization

This repository contains the original implementation of the face normalization method Fan-C descibed in the following paper.
Please cite the paper if you use this code and/or our trained models.

> P. Werner, F. Saxen, A. Al-Hamadi, and Hui Yu, ["Generalizing to Unseen Head Poses in Facial Expression Recognition and Action Unit Intensity Estimation"](https://www.researchgate.net/publication/332979114), IEEE International Conference on Automatic Face and Gesture Recognition (FG), 2019.

In the paper the method is called "FaNC", but we like to pronounce it as "fancy" and that is why we prefer to write it as "Fan-C" now.
The acronym stands for "**Fa**ce **N**ormalization based on Learning **C**orrespondences".
The key idea is to predict coordinates and visibilities of correspondence points from facial landmarks in both the arbitrary-pose source domain and the frontal target domain.
The predicted information is used to generate a face image that is normalized regarding pose and facial proportions.
Fan-C can be learned and applied on top of any landmark localizer, also without facial contour landmarks, and is very fast.
With our unpublished OpenGL implementation it runs in less than 2 ms even on old and cheap on-board GPUs.
You can find more details in the [paper full-text](https://www.researchgate.net/publication/332979114) or get a more detailed overview through our [poster](https://www.researchgate.net/publication/333209907).

## Requirements

To run the code you need:

- Matlab (tested with 2014a and some newer versions)
- Our SyLaFaN dataset and/or our pre-trained models

## Getting started

**Caution: Please note that the SyLaFaN dataset and our pre-trained models are only available for non-commercial research purposes, because they are based on a non-commercial academic licence of [FaceGen](https://facegen.com/).**

1. Download/unpack or clone this repository
2. Download the [SyLaFaN Dataset](http://wasd.urz.uni-magdeburg.de/pwerner/fan-c/DB_SyLaFaN.zip)
3. Download the [pre-trained Fan-C models](http://wasd.urz.uni-magdeburg.de/pwerner/fan-c/fan-c-models.zip)
4. Extract the contents of the ZIP files into the root directory of this repository
5. Run the scripts in the root directory

## Note

**More code will be added soon.**



