# Surge Sense

to hyper parameter training run python hyperparameter_tuning RandomForest - for tuning randomforest 
to hyper parameter training run python hyperparameter_tuning RandomForest - for tuning XGBoost 
to hyper parameter training run python hyperparameter_tuning RandomForest - for tuning GradientBoosting 

# MLOps-Driven Machine Learning Project on EC2

This repository demonstrates the implementation of MLOps principles in the development, deployment, and monitoring of a machine learning application. The project integrates automation, continuous integration/continuous deployment (CI/CD), and scalability to ensure efficient and reliable ML workflows.

##  Features:
- **Dataset**: Wine Quality dataset 
    - dataset like :- https://www.kaggle.com/datasets/yasserh/wine-quality-dataset  
- **Model**: ElasticNet
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
https://github.com/Immortal-Pi/ML-project-with-MLFlow
```

### STEP 01- Create a conda environment after opening the repository
```bash 
conda create -n mlopscnn python=3.9 -y
```
```bash 
conda activate mlopsML
```

### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```

# Finally run the following command
```bash
python app.py
```

Now,

open up you local host and port


# MLflow & pipeline tracking

dagshub repo : https://dagshub.com/Immortal-Pi/ML-project-with-MLFlow 

```bash
import dagshub
dagshub.init(repo_owner='your-github-username', repo_name='your-repository-name', mlflow=True)
```


### mlflow experiments 
```bash 
- mlflow ui 
``` 
![MLFlow Workflow](https://github.com/Immortal-Pi/ML-project-with-MLFlow/blob/main/documentation/mlflow1.png)

![MLFlow Workflow](https://github.com/Immortal-Pi/ML-project-with-MLFlow/blob/main/documentation/mlflow2.png)


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

# required 
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu 
newgrp docker 
```
### 6. Configure EC2 as self-hosted runner: 
``` bash 
settings> actions> runner>new self hosted runner > choose os > then sun the command one by one 
```
### 7. setup github secrets 
``` bash
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION = us-east-1
AWS_ECR_LOGIN_URI = demo>>  566373416292.dkr.ecr.ap-south-1.amazonaws.com
ECR_REPOSITORY_NAME = simple-app
```

## Demo 

![Demo of ML Project](https://github.com/Immortal-Pi/ML-project-with-MLFlow/blob/main/documentation/demo.gif)



## Tech Stack 

- **Programming Language**: Python
- **Deep Learning Framework**: Keras with TensorFlow backend
- **MLOps Tools**:
    - Docker for containerization
    - Github actions CICD pipelines
- **Web Framework**: Flask for model deployment
- **Cloud Platform**: AWS for hosting the model
- **Version Control**: Git and GitHub
- **Data Utilities**: YAML, JSON handling, and custom preprocessing functions

## Conclusion
This project highlights the integration of MLOps principles in managing the entire machine learning lifecycle. While the focus was on building a wine quality prediction model using regression techniques, the core objective was to emphasize the importance of project structure, automation of workflows, and the use of tools like Docker for deployment. Additionally, a CI/CD pipeline was implemented to automate the testing, building, and deployment processes, ensuring consistent and reliable updates to the application. This project serves as a foundation for understanding how to design scalable, maintainable, and efficient ML pipelines, ensuring reproducibility and streamlined collaboration in real-world scenarios.



