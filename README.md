# How to use
- install python
- install git
- open CMD in an empty folder folder
- clone this repo `git clone https://github.com/mohandabc/Doc_rebuilt .`
- create a virtual environement in the folder `python -m venv .venv`
- activate the environement `.venv\Scripts\activate`(on windows) or execute a.but script
- install requirements `pip install -r requirements.txt`
- edit tests.py file, uncomment the test you want to perform (train_CNN, segmentation, etc)
- run `python tests.py`

# Dataset
Dataset used in this project are composed of windows (or portions) of lesion images. In order to avoid computing the windows again and again each time we need them to train a model, we decided to generate these datasets. For this we take an ordinary dataset like ISISC 2017 and we run `create_dataset`function in `tests.py` file giving in the source dataset. This will generate two datasets in a standard form, one with windows of size (31,31), the other (63, 63). Having two datasets is a requirement of the model we created. We can generate more than 2 level by editing `Dataset.py`.

Dataset used here is not included as it is too large. But it can be generated using ISIC2017 training dataset.
