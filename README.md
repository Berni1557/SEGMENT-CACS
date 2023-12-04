# SEGMENT-CACS
Multi-Task deep learning model for segment-level coronary artery calcium scoring on non-contrast CT

This repository is the official implementation of [Automated segment-level coronary artery calcium scoring on non-contrast CT: multi-task deep learning approach](https://arxiv.org/abs/2030.12345).


![Multi-task Network structure](images/multi_task_network.png)

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Prediction

To predict the model in the paper, copy  all non-contrast enhanced cardiac CT (.mhd file format) into the data folder. To run the script, you can run the cacs.sh script or run the python script in the terminal directly.

```
python cacs_predict.py -m <path_to_pretrained_model> -d <path_to_data_folder> -p <path_to_prediction_folder> -gpu cuda
```
### Example prediction
<img src="images/prediction_github.png" alt="drawing" width="500"/>

## Pre-trained Models

You can find the pretrained model in the model folder.

## Training

Since the training set and corresponding annotations are not public, a training script is not provided. 

## Contributing

Bernhard Föllmer  
Charité - Universitätsmedizin Berlin  
Klinik für Radiologie  
Campus Charité Mitte (CCM) 
