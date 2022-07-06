# {{cookiecutter.project_name}}

{{cookiecutter.description}}

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

## Training on FashionMNIST 

This particular template contains all the code required to train a simple Convolutional Neural Network (CNN) on the FashionMNIST classification task.
It shows the use of PyTorch, PyTorch Lightning, and Hydra. As such, it can serve as a guide for your own project.
To get started, create the appropriate conda environment:

```bash
conda env create -n fmnist -f environment.yaml
```

Then train the simple CNN on the FashionMNIST classification task:

```bash
python train.py train=fashionmnist
```

### Troubleshooting

- If the conda environment creation process complains about CUDA, please comment out the `cudatoolkit` requirement from `environment.yaml`.
- If training fails due to a GPU not being found, try out the `cpu` accelerator by modifying `configs/train/fashionmnist.yaml`.