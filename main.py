import os
import io 
import json 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torchvision import models,transforms 
from PIL import Image 
import time
from flask import jsonify


# lazy global 
device  = None
model = None
imagenet_class_index = None 


def img_to_tensor(image_bytes):
    """Converts byte arrya to torch.tensor with transforms
    
    Args:
    -----
        img: byte
            input image as raw bytes 
    
    Returns: 
    --------
        img_tensor: torch.Tensor
            image as Tensor for using with deep learning model
    """

    # transformations for raw image
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    
    img = Image.open(io.BytesIO(image_bytes))
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor.to(device)


def get_prediction(image_bytes):
    """perform predictions using model defined globally
    
    Args:
    -----
        image_bytes:bytes
            raw image bytes recieved via POST
    
    Returns:
    --------
        class_id: int
            id defined in imagenet_class_index.json
        class_name: str
            top predicted category 
        prob: float
            confidence score for prediction    
    """
    tensor = img_to_tensor(image_bytes=image_bytes)
    outputs = F.softmax(model.forward(tensor),dim=1)
    prob, y_hat = outputs.max(1)
    prob = prob.item()
    predicted_idx = str(y_hat.item())
    class_id, class_name = imagenet_class_index[predicted_idx]
    return class_id, class_name, prob


def handler(request):
    """Entry point for cloud function

    Args:
    -----
        request: Flask.request
            contains incoming data via HTTP POST
    
    Return:
    -------
        inference results as Flask.jsonify object
    """
    global device, model, imagenet_class_index
    if device is None:
        print("device created")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )
    if model is None:
        print("creating resnet50 model")
        model = models.resnet18(pretrained=True)
        model.eval()
        model.to(device)
    if imagenet_class_index is None: 
        print("loading imagenet class names ")
        imagenet_class_index = json.load(open('imagenet_class_index.json'))

    if request.method=='POST':
        print("postrequest received")
        file = request.files['file']        
        img_bytes = file.read()
        class_id, class_name, prob = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_id': class_id, 'class_name': class_name})
    else:
        return "Please specify image"
        