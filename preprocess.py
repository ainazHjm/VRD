# from torchvision import models, transforms
from PIL import Image
from torch.autograd import Variable

import numpy as np
import csv
import torch

img_size = 224

def abnormal(coordinates, width, height):
    coordinates[0] = coordinates[0] * width
    coordinates[1] = coordinates[1] * width
    coordinates[2] = coordinates[2] * height
    coordinates[3] = coordinates[3] * height
    return coordinates

def get_bbox(img_id, coordinates):
    id_bbox = {}
    try:
        img = Image.open('/home/mehran/Desktop/workspace/get images/images/' + img_id + '.jpg')
        # img = Image.open('../data/sampled/' + img_id + '.jpg')
        width, height = img.size
        img = img.convert('RGB')

        for i in range(len(coordinates)):
            coords = list(map(float, coordinates[i]))
            coords = abnormal(coords, width, height)
            id_bbox.setdefault(img_id, [])
            bbox = np.array(img.crop((coords[0], coords[2], coords[1], coords[3])).resize((img_size,img_size)))
            id_bbox[img_id].append(bbox.reshape((1, 3, img_size, img_size)))
            # id_bbox[img_id].append(Variable(torch.FloatTensor(bbox)).cuda().view(3, img_size, img_size).unsqueeze(0))
            # id_bbox[img_id].append(Variable(torch.FloatTensor(bbox)).view(3, img_size, img_size).unsqueeze(0))
    
    except(OSError):
        print('----- image not available')
        pass

    return id_bbox

def prepare_data(attr_lables):
    img_id = []
    id_bbox_dict = {}
    id_labels = {}

    with open('challenge-2018-train-vrd.csv') as train_vrd:
        f = csv.reader(train_vrd, delimiter = ',')
        line_number = 0
            
        for row in f:        
            if line_number > 0:
                if row[-1] == 'is':
                    img_id.append(row[0])
                    id_bbox_dict.setdefault(row[0], [])
                    id_bbox_dict[row[0]].append(row[3:7])
                    id_labels.setdefault(row[0], [])
                    id_labels[row[0]].append(attr_lables[row[2]])
            
            line_number = line_number + 1

    return img_id, id_bbox_dict, id_labels
