# Moodylyser
 Emotion detector project that employs Machine Learning


<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Tech Stack](#tech-stack)
  * [File Structure](#file-structure)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [Results and Demo](#results-and-demo)
* [Future Work](#future-work)
* [Troubleshooting](#troubleshooting)
* [Contributors](#contributors)
* [Acknowledgements and Resources](#acknowledgements-and-resources)
* [License](#license)


<!-- ABOUT THE PROJECT -->
## About The Project
<!-- [![Product Name Screen Shot][product-screenshot]](https://example.com)   -->

Aim and Description of project.  
<!-- Refer this [documentation](https://link/to/report/) -->

### Tech Stack
This section should list the technologies you used for this project. Leave any add-ons/plugins for the prerequisite section. Here are a few examples.
* [Keras](https://keras.io/)
* [TensorFlow](https://www.tensorflow.org/)
* [Python](https://www.python.org/)
* [OpenCV](https://opencv.org/)
* [Matplotlib](https://matplotlib.org/)
* [Numpy](https://numpy.org/doc/#)  
* [Google Colab](https://colab.research.google.com/)
* [Kaggle Notebooks](https://www.kaggle.com/notebooks)
* [Kaggle Dataset(Facial expression challenge)](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)


### File Structure
    .
    ├── app.py                  # Explain the function preformed by this file in short
    ├── docs                    # Documentation files (alternatively `doc`)
    │   ├── report.pdf          # Project report
    │   └── results             # Folder containing screenshots, gifs, videos of results
    ├── src                     # Source files (alternatively `lib` or `app`)
    ├── ...
    ├── test                    # Test files (alternatively `spec` or `tests`)
    │   ├── benchmarks          # Load and stress tests
    │   ├── integration         # End-to-end, integration tests (alternatively `e2e`)
    │   └── unit                # Unit tests
    ├── ...
    ├── tools                   # Tools and utilities
    ├── LICENSE
    ├── README.md
    ├── Setup.md                # If Installation instructions are lengthy    
    └── todo.md                 # If Future developments and current status gets lengthy


<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

* See [SETUP.md](https://link/to/setup.md) if there are plenty of instructions
* The installations provided below are subjective to the machine your are using
* We used [pip install(https://pip.pypa.io/en/stable/)] for the installations. If you don't have pip please follow the following command
```sh
 python3 -m pip install -U pip
```
* List of softwares with version tested on:
  * TensorFlow
  ```sh
   python3 -m pip install tensorflow
  ```
  * Numpy
  ```sh
   python3 -m pip install numpy
  ```
  * dlib
  ```sh
   pip install cmake
   pip install dlib
  ```
  * Download the Shape predictor file from [here(https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat)] and insert it in your current project folder
  * Matplotlib
  ```sh
   python3 -m pip install matplotlib
  ```
  * OpenCV
  ```sh
   python3 -m pip install opencv-contrib-python
  ```
  * scikit learn


* **ESP-IDF v4.0 and above**

  You can visit the [ESP-IDF Programmming Guide](https://docs.espressif.com/projects/esp-idf/en/latest/get-started/index.html#installation-step-by-step) for the installation steps.

* See if you can make requirements.txt  
```sh
pip install -r "requirements.txt"
```

### Installation
1. Clone the repo
```sh
git clone https://github.com/hashmis79/Moodylyser
```


<!-- USAGE EXAMPLES -->
## Usage
* After cloning the repo transfer the files to your project folder. Open terminal and go to the project folder and run the following commands
```sh
cd .../projectfolder
python3 model.py
```


<!-- RESULTS AND DEMO -->
## Results and Demo
Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space.  
[**result screenshots**](https://result.png)  
![**result gif or video**](https://result.gif)  

| Use  |  Table  |
|:----:|:-------:|
| For  | Comparison|


<!-- FUTURE WORK -->
## Future Work
* See [todo.md](https://todo.md) for seeing developments of this project
- [x] To Make an emotion detector model
- [x] To connect it to a live feed for live detection
- [x] To give statistical data in the form of graphs
- [ ] To increase the accuracy of the model
- [ ] To deploy the model in the form of an emotion detector app or site


<!-- TROUBLESHOOTING -->
## Troubleshooting
* Common errors while configuring the project


<!-- CONTRIBUTORS -->
## Contributors
* [Anushree Sabnis](https://github.com/hashmis79)
* [Saad Hashmi](https://github.com/id)
* [Shivam Pawar](https://github.com/theshivv)
* [Vivek Rajput](https://github.com/Vivek-RRajput)


<!-- ACKNOWLEDGEMENTS AND REFERENCES -->
## Acknowledgements and Resources
* [SRA VJTI](http://sra.vjti.info/) Eklavya 2020  
* Refered [this](https://www.coursera.org/learn/introduction-tensorflow) for understanding how to use tensorflow
* Refered [this](https://www.coursera.org/learn/convolutional-neural-networks) course for understanding Convolutional Neural Networks
* Refered [towardsdatascience](https://towardsdatascience.com/) and [machinelearningmastery](https://machinelearningmastery.com/) for frequent doubts  
...


<!-- LICENSE -->
## License
Describe your [License](LICENSE) for your project.
