# torch-gcp-fn
Deploy Image classification on GCP as cloud function.

## About Cloud Function
Cloud function has good response time and ability to scale. These can be configured accordingly using `requirements.txt`. 
For more understanding, this [post](https://cloud.google.com/blog/products/ai-machine-learning/how-to-serve-deep-learning-models-using-tensorflow-2-0-with-cloud-functions) describes how TF is used for deploying classification model. 

# App
This app uses imagenet pre-trained model `resnet-18` available in `torchvision` to give image classification according to 1000 predefined categories in `imagenet_class_index.json`.

# Setup1: local machine 
- clone this repo
- cd `$REPO;zip -r torch_gcp_fn.zip . ` 

This will make a zip file of this repo. 


# Setup2: Cloud Function

- On GCP console, go to `Cloud Functions` and click `Create Function`.
- Fill `name` as desired, add memory `2GB`, set Triger `HTTP`.
- Under `Source code`, select `Zip upload`.
- Select Runtime as `Python3`
- set `functions to execute` to  `handler`
- Expand `Environment variables, networking, timeouts and more`
- set `Timeout` to max  as `540`
- click create


# Response
- once the function is launched correctly, there will be green tick 
- to start inference and testing, run following with updated values 

```python
import requests

function_url = "" # add url path to cloud function previously created
file_path = "" # add local image path 
resp = requests.post(function_url, files={"file":open(file_path,'rb')})
print(resp.json())
```







