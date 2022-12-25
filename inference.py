from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as T

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch.nn.functional as F
import os
import numpy as np
import pickle
import heapq


def foodclassification(img_path):
    device = "cuda"
    torch.set_grad_enabled(False)
    transform = T.Compose([
        T.Resize(256, interpolation=3),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])

    with open('outputs/cub200/type.pkl', 'rb') as f:
        type = pickle.load(f)
    model = torch.load('outputs/cub200/encoder.pth')
    embedding = torch.load('outputs/cub200/embedding.pt')
    model.eval()
    model = model.to(device)
    embedding = torch.FloatTensor(embedding)
    embedding = embedding.squeeze(1)
    images = Image.open(img_path)
    images = transform(images).unsqueeze(0)
    images = images.to(device)

    output = model(images)
    #print(model)
    if isinstance(output, tuple):
        output = output[0]
    features = F.normalize(output, dim=1)
    features = features.t()
    prob = torch.mm(embedding,features.cpu())
    prob = prob.tolist()
    heap = list(prob)
    heapq.heapify(heap)
    topn = 10
    largest = heapq.nlargest(topn, heap)
    indices = [prob.index(x) for x in largest]
    count = 0
    answer = []
    for i in range(topn):
        if largest[i][0]>0.6:
            answer.append(type[indices[i]])
            count = count+1
    answer_set = set(answer)
    answer = list(answer_set)
    return answer    
    if count==0:
        return ["找不到您的食物"]
