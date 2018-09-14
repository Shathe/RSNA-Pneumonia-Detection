# RSNA Pneumonia Detection
This repository aims to detect RSNA Pneumonia and participate on the [RSNA Pneumonia Detection kaggle challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) 

## Data
Download the data from the [kaggle challenge page](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data). Extract the folder on the root directory and rename it as 'data'.
Then, change add read permissions to all the files inside it. Extract the zip files. 

## Requirements

```
pip install requirements.txt
```

## Data exploratory
```
python data_exploratory.py
```

## To do
implementate the [semantic segmentation](algorithms/semanticSegmentation) and [object detection](algorithms/objectDetection) algorithms.
With the ebst option, try to add aditional information like gender, age...