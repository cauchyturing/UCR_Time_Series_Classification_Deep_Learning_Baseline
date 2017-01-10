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



## MLP

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing

A step by step series of examples that tell you have to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone who's code was used
* Inspiration
* etc
