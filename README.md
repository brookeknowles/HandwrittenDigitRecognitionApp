<div align="center" id="top">
    <h1>Digit Recogniser</h1>
    <h5>GUI and model for digit recognition</h5>
    <p>
        <b>Digit Recogniser</b> is a GUI to experiment with a digit recognition machine learning model. 
        This project was created for the <b>COMPSYS 302</b> course at the <b>University of Auckland</b>.
    </p>
    <a href="https://python.org">
        <img alt="Python 3.9" src="https://img.shields.io/badge/python-3.9-informational.svg?style=flat-square&logo=python&logoColor=white">
    </a>
    <a href="https://github.com/COMPSYS-302-2021/project-1-team_19/actions/workflows/python-app.yml">
        <img alt="GitHub Action" src="https://github.com/COMPSYS-302-2021/project-1-team_19/actions/workflows/python-app.yml/badge.svg">
    </a>
</div>

# Table Of Contents
*   [Features](#features)
*   [Quick Start](#quick-start)
*   [Usage](#usage)
*   [Roadmap](#roadmap)
*   [Authors](#authors)
*   [License](#license)
*   [Status](#status)

# Features
* Easy model training
* Dataset download
* Choice of different algorithms to recognise digits

[Back To Top](#top)

# Quick Start
1. Clone the repository
```shell
git clone https://github.com/COMPSYS-302-2021/project-1-team_19.git
```

2. Create a virtual environment
    1. Make sure you are using python 3.9 with `python --version`
    2. Navigate into the correct folder: `cd project-1-team_19`
    3. Run `python -m venv venv`
    4. Activate with `source venv/bin/activate` (may be different depending on your os)
    
3. Install Requirements
```shell
pip -r install requirements.txt
```
4. Run
```shell
python main.py run dataset
```

[Back To Top](#top)

# Usage
### Creating a Dataset
1. Run the dataset creation tool
```shell
./main.py run dataset
```
2. Draw the digit shown on the right and click submit
3. Repeat until you have enough images
4. Export the dataset

### Validating a Dataset

### Digit Recognition
1. Run the recognition application
```shell
./main.py run recognition
```
2. Draw any digit in the canvas
3. Choose <kbd>Model</kbd> to get accurate results
4. Click <kbd>Recognise</kbd>

[Back To Top](#top)

# Roadmap
* [All issues are on Github](https://github.com/COMPSYS-302-2021/project-1-team_19/issues)
* [Roadmap is on Github Project](https://github.com/COMPSYS-302-2021/project-1-team_19/projects/1)

# Authors
| ![j-chad](https://avatars.githubusercontent.com/u/7777317?v=4&s=100) | ![brookeknowles](https://avatars.githubusercontent.com/u/62309663?v=4&s=100) |
|:--------------------------------------------------------------------:|:----------------------------------------------------------------------------:|
|                           Jackson Chadfield                          |                                Brooke Knowles                                |
|                  [Github](https://github.com/j-chad)                 |                  [Github](https://github.com/brookeknowles)                  |
|            [Email](mailto:jackson.chadfield@jacksonc.dev)            |                   [Email](mailto:bkno906@aucklanduni.ac.nz)                  |

# License
This project is currently not open source, and should not be distributed to unauthorised users.

# Status
This project is currently in a finished state. 
Further maintenance is unlikely.
<hr>

[Back To Top](#top)
