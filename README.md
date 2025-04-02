# Surge Sense - An ML prediction model for surge price 

The goal of this project was to build a full end-to-end machine learning pipeline to predict surge pricing for cab rides, specifically comparing Uber and Lyft services in the New York City area. The pipeline aimed to leverage historical ride, weather, and event data to forecast price surges based on factors such as time, location, weather, and demand patterns. As part of the modeling phase, we utilized multiple machine learning algorithms—including Random Forest, XGBoost, and Gradient Descent-based models—and selected the best-performing model for accurate surge price prediction.


# MLOps-Driven Machine Learning Project on EC2

This repository demonstrates the implementation of MLOps principles in the development, deployment, and monitoring of a machine learning application. The project integrates automation, continuous integration/continuous deployment (CI/CD), and scalability to ensure efficient and reliable ML workflows.

##  Features:
- **Dataset**: Uber & Lyft dataset  
    - dataset:- https://www.kaggle.com/datasets/ravi72munde/uber-lyft-cab-prices
- **Model**: XGBoost
- **MLOps Integration**:
    - Config-driven pipeline setup with YAML.
    - Data versioning and workflow tracking using DVC.
    - Modular project structure with reusable components.
- **Deployment**: Dockerized Flask app for serving the model and deployed on AWS.
- **Utilities**: Common functions for reading YAML, saving JSON, and handling data.
- **Project Workflow**:
    - Data ingestion and preprocessing.
    - Model training and evaluation.
    - Deployment with Docker and AWS.

# How to run?

### STEPS:
Clone the repository
```bash
https://github.com/Immortal-Pi/SurgeSense.git
```

### STEP 01- Create a conda environment after opening the repository
```bash 
conda create -p venv python==3.10 -y
```
```bash 
conda activate venv/
```

### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```
### STEP 03- train the model 
```bash
python main.py 
```
### Optional - for HyperParameter Tuning 

RandomForest - python hyperparameter_tuning.py RandomForest
XGBoost - python hyperparameter_tuning.py XGBoost
Gradient Boosting - python hyperparameter_tuning.py GradientBoosting


# Finally run the following command
```bash
python app.py
```


Now,

open up you local host and port


# MLflow & pipeline tracking

dagshub repo : https://dagshub.com/Immortal-Pi/SurgeSense

```bash
import dagshub
dagshub.init(repo_owner='your-github-username', repo_name='your-repository-name', mlflow=True)
```


### mlflow experiment tacking and DVC pipeline tracking 

MLFlow - experiment tracking 

![MLFlow Workflow](https://github.com/Immortal-Pi/ML-project-with-MLFlow/blob/main/documentation/mlflow1.png)

![MLFlow Workflow](https://github.com/Immortal-Pi/ML-project-with-MLFlow/blob/main/documentation/mlflow2.png)

DVC - pipeline tracking 

```bash 
dvc init 
dvc repro 
dvc dag 
```
![DVC pipeline ](https://github.com/Immortal-Pi/ML-project-with-MLFlow/blob/main/documentation/mlflow2.png)

## AWS CICD Deployment with Github Actions 

### 1. Login to AWS console

### 2. Create IAM user for deployment 

``` bash 
# with access 
1. ECR access : It is vurtual machine 
2. ECR: Elastic Container registry to save your docker image in AWS

# Description: About the deployment 
1. Build docker image of the source code 
2. Push your docker image to ECR
3. Launch Your EC2
4. Pull your image from ECR in EC2
5. Launch your docker image in EC2

# Policy 
1. AmazonEC2ContainerRegistryFullAccess
2. AmazonEC2FullAccess
```
### 3. Create ECR repo to store/save docker image 
``` bash
- Save the URI: 011528265658.dkr.ecr.us-east-1.amazonaws.com/mlproj
```
### 4. Create EC2 machine (Ubuntu)
### 5. Open EC2 and install docker in EC2 Machine:
```bash
# optional 
sudo apt-get update -y
sudo apt-get upgrade -y 

# required to install dockers 
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu 
newgrp docker 
```
### 6. Configure EC2 as self-hosted runner: 
``` bash 
settings -> actions -> runner -> new self hosted runner -> choose os -> then run the command one by one on the EC2 console
```
### 7. setup github secrets 
``` bash
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION = us-east-1
AWS_ECR_LOGIN_URI = example >>  566373416292.dkr.ecr.ap-south-1.amazonaws.com
ECR_REPOSITORY_NAME = simple-app
```

## Demo 

![Demo of ML Project](https://github.com/Immortal-Pi/ML-project-with-MLFlow/blob/main/documentation/demo.gif)



## Tech Stack 

- **Programming Language**: Python
- **Hyperparamter Tuning**: GridSearchCV and HyperOpt
- **ML algorithms**: Random Forest, XGBoost, Gradient Boosting 
- **MLOps Tools**:
    - Docker for containerization
    - Github actions CICD pipelines
- **Web Framework**: Flask for frontend application 
- **Cloud Platform**: AWS EC2 for hosting the application
- **Version Control**: Git and GitHub
- **Data Utilities**: YAML, JSON handling, and custom preprocessing functions

## Conclusion

The implemented pipeline successfully automated data collection, preprocessing, feature engineering, model training, and prediction. Among the models tested, XGBoost outperformed others based on key evaluation metrics including Root Mean Squared Error (RMSE), Mean Squared Error (MSE), and R-squared (R²) score. As a result, the XGBoost model was used for final predictions. The deployed system provided reliable, real-time surge pricing forecasts, helping users make informed decisions when choosing between Uber and Lyft in New York City.



