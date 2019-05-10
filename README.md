# Fan-C Face Frontalization

This repository contains the original implementation of the face normalization method Fan-C descibed in the following paper.
Please cite the paper if you use this code and/or our trained models.

> P. Werner, F. Saxen, A. Al-Hamadi, and Hui Yu, ["Generalizing to Unseen Head Poses in Facial Expression Recognition and Action Unit Intensity Estimation"](https://www.researchgate.net/publication/332979114), IEEE International Conference on Automatic Face and Gesture Recognition (FG), 2019.

In the paper the method is called "FaNC", but we like to pronounce it as "fancy" and that is why we prefer to write it as "Fan-C" now.
The acronym stands for "**Fa**ce **N**ormalization based on Learning **C**orrespondences".
The key idea is to predict coordinates and visibilities of correspondence points from facial landmarks.
The predicted information is used to generate a face image that is normalized regarding pose and facial proportions.
FaN-C can be learned and applied on top of any landmark localizer, also without facial contour landmarks, and runs in less than 2 ms even on cheap on-board GPUs.

**The code, trained models etc. will be added a few days after the conference (between 20th and 24th of May, 2019).**



