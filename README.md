# sagemaker-dogbreeds-classification
This project aims to use AWS Sagemaker to implement a full MLOps pipeline. The machine learning problem is very simple: dog breeds classificaion (133 classes). The original data is from [here](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip). For this project, I've cleaned it and uploaded it to my S3 bucket `s3://{BUCKET}/all/`. `BUCKET` need to be provided as an environment variable before running `pipeline.ipynb`. 

This project implements the following in Sagemaker:
- a training pipeline
- a deployment pipeline
- a data monitoring and a model monitoring schedule

![training and deployment pipelines](images/deployment.png)

![data and model monitoring](images/monitoring.png)

The images are from [the ml.school program from Santiago](https://github.com/svpino/ml.school).
The pipeline constructed here is a little bit different:
- there is no "inference pipeline", I put pre-processing and post-processing of the model in the same script using `input_fn` and `output_fn`. Necessary artifacts that should always be together with model checkpoint is directly put inside `SM_MODEL_DIR` in the training job, in this case the classes name and order of the data, but in general could be some artifacts for pre-processing transformation. 
- there is no batch transform job for generating predictions for test data, it is done in the evaluation step. As I see in evaluation step, predictions for testing data is already made, I directly save those prediction and use them as the model performance baseline for later model monitoring. 

Main code is in `pipeline.ipynb`. Other scripts are used in various AWS or Sagemaker services, e.g. training jobs, endpoints, monitoring, lambda. 

The notebook can be run as is, the only config to supply is the `BUCKET` environment variable. It can be put in `config.env` or `.env` (need code changes in the first 2 cells). `BUCKET` provide a S3 bucket that can be used for this project, artifacts from various steps (e.g. training, evaluation) in the pipeline woule be output to this bucket as well as data capturing from the deployed endpoints and its monitoring. `BUCKET` is just the name without the `s3://`. 
- I have also put my raw image data in `s3://{BUCKET}/all/`, with 133 subfolders inside and each subfolder name being a dog breed and containing images of the respective breed of dogs.

The training pipeline is constructed directly in the `.ipynb`. 
![training pipeline in Sagemaker](images/pipeline.png)

The deployment pipeline is done via AWS console UI as I don't have the permission in AWS Sagemaker correcly configured. 

For the monitoring schedule, some fake traffic and groundtruth must be generated first for any statistics to capture and show in the monitoring schedule. The last few cells demonstrated the code. 

Monitoring schedule demo: 
![sagemaker monitoring](images/sagemaker_monitoring.png)

# Reference
this repo is a class project from the awesome program: [ML School program from Santiago](https://github.com/svpino/ml.school)
