A common repo for Classifier

DATASET:
    To make dataloader the images should be divided into the folders each of every class in the dataset. Do this for train, validation and test dataset.
    
TRAIN:

    Use the train_classfier.py file to train your classifier. Modify the ./config/config.yaml file accordingly and use the following command to train your classifier
    
    "CUDA_VISIBLE_DEVICES=0 python train_classifier.py --config "./config/config.yaml""

AVAILABLE MODELS : 
    Resnet (plus its variants), Efficientnet (plus its variants) and mobilenet
    To add more pretrained models, modify the models/classifier.py file to include for models from torchvision or from other repository.

INFERENCE

    predict_class.py file provides the inference steps.
    Set the class names accordingly on which you trained the network. Also set the mean and std deviation for your dataset and other parameters in the config_predict.yaml file.
    
Note: early_steps parameter sets the number of epoch to test for early stopping
