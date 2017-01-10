# Deep Learning for Time Series Classification 
As the simplest type of time series
data, univariate time series provides a reasonably good starting
point to study the temporal signals. The representation
learning and classification research has found many potential
application in the fields like finance, industry, and health care. Common similarity measures like Dynamic Time Warping (DTW) or the Euclidean Distance (ED) are decades old. Recent efforts on different feature engineering and distance measures designing give much higher accuracy on the UCR time series classification benchmarks (like BOSS [[1]](http://link.springer.com/article/10.1007%2Fs10618-015-0441-y),[[2]](http://link.springer.com/article/10.1007%2Fs10618-014-0377-7), PROP [[3]](http://link.springer.com/article/10.1007/s10618-014-0361-2) and COTE [[4]](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7069254)) but also let to the pitfalls of higher complexity and interpretability. 

The exploition on the deep neural networks, especially
convolutional neural networks (CNN) for end-to-end time
series classification are also under active exploration like multi-channel CNN (MC-CNN) [[5]](http://link.springer.com/article/10.1007/s11704-015-4478-2)
and multi-scale CNN (MCNN) [[6]](https://arxiv.org/abs/1603.06995). However, they still need heavy preprocessing and a large set of hyperparameters which would make
the model complicated to deploy. 

This repository contains three deep neural networks models (MLP, FCN and ResNet) for the pure end-to-end and interpretable time series analytics. These models provide a good baseline for both application for real-world data and future research in deep learning on time series.



## Network Structure
![Picture] (https://www.dropbox.com/s/mh46aoliorpdbl9/Archi.pdf)
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone who's code was used
* Inspiration
* etc
