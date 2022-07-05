# Cookiecutter DNEG ML

A humble branch of the extensive template [cookiecutter-data-science](http://drivendata.github.io/cookiecutter-data-science/).
This particular template is also fully functioning out of the box. 
It contains everything to train a very simple Convolutional Neural Network (CNN) on the FashionMNIST classification task.
It uses PyTorch, PyTorch Lightning, and Hydra. As such, it can serve as a guide for your own project.

### Requirements to use the cookiecutter template:
-----------
 - Python 3.7+
 - [Cookiecutter Python package](http://cookiecutter.readthedocs.org/en/latest/installation.html): This can be installed with pip by or conda depending on how you manage your Python packages:

``` bash
$ pip install cookiecutter
```

or

``` bash
$ conda config --add channels conda-forge
$ conda install cookiecutter
```


### To start a new project, run:
------------

    cookiecutter -c v1 https://github.com/drivendata/cookiecutter-data-science


### The resulting directory structure
------------

The directory structure of your new project looks like this: 

```
├── LICENSE
├── README.md               <- The top-level README.
├── data                    <- Store limited amount of useful data here. NO CLIENT IP.
├── docs                    <- Document project here. It is recommended to use Markdown.
│
├── models                  <- Trained and serialized models
│
├── notebooks               <- Jupyter notebooks.
│
├── requirements.txt        <- The requirements file.
│
├── setup.py                <- Project can be made pip installable
├── src                     <- Source code for use in this project.
│   ├── __init__.py         <- Makes src a Python module
│   │
│   ├── data                <- All dataset related code
│   │   └── dataset.py      <- PyTorch compatible dataset
│   │   └── data_module.py  <- PyTorch Lightning data module
│   │
│   ├── models              <- All model related code
│   │   ├── model.py        <- Contains an example PyTorch model.
│   │   └── model_module.py <- PyTorch Lightning model module
│   │
│   └── visualization       <- All visualization code
│       └── visualize.py
```

### Installing development requirements
------------

    pip install -r requirements.txt

