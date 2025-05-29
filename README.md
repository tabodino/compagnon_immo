Compagnon Immo
==============================

This repository presents a case study on price prediction for the real estate market, as well as the analysis of price trends over time

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Should be in your computer but not on Github (only in .gitignore)
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's name, and a short `-` delimited description, e.g.
    │                         `1.0-alban-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, links, and all other explanatory materials.
    │
    ├── reports            <- The reports that you'll make during this project as PDF
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   └── visualize.py

--------

## Setup environment

With conda

1. Create the vitual environment
```
conda create -f environment.yml
```
2. Activate the environment
- Linux / MacOS
```
source activate comp_immo_env
```
- Windows
```
conda activate comp_immo_env
```

With pip

1. Create the vitual environment
- Linux / MacOS
```
python3 -m venv venv
source venv/bin/activate
```
- Windows
```
python -m venv venv
venv\Scripts\activate
```
2. Install dependencies
```
pip install -r requirements.txt
```



## Extract, transform and load data
> Depending on your system, use either python3 (Linux/macOS) or python (Windows) when running Python commands in the terminal.

<p>To apply ETL, run this command:</p>

<p>For all data (all years and all departments):</p>

```bash
python src/data/make_dataset.py all
```
<p>For all departments on a specific year :</p>

```bash
python src/data/make_dataset.py year --year 2024
```

<p>For all years on a specific department :</p>

```bash
python src/data/make_dataset.py dep_all --dep 75
```
<p>For a department in a specific year:</p>

```bash
python src/data/make_dataset.py dep --dep 75 --year 2024
```

----
## Train and predict model

Run train script:
```
python src/models/train_model.py
```

Run predict script:
```
python src/models/predict_model.py
```

## Launch streamlit app

```
streamlit run src/streamlit/app.py
```

----
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
