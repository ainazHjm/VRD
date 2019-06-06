import preprocess
import train_iters

import torch.nn as nn
import numpy as np
import torch

from pathlib import Path
from random import shuffle
from torch.autograd import Variable

def is_eq(pred_indices, label_vec):
    for idx in pred_indices:
        if label_vec[idx] == 1:
            return 1
    return 0

attr_lables = {
    '/m/02gy9n': 0, #Transparent
    '/m/05z87': 1, #Plastic
    '/m/0dnr7': 2, #(made of)Textile
    '/m/04lbp': 3, #(made of)Leather
    '/m/083vt': 4 #Wooden
}

img_id, id_bbox_dict, id_labels = preprocess.prepare_data(attr_lables)
feature_path = Path('features/')
feature_path_list = list(feature_path.glob('*.npy'))

feature_path_names = list(range(0,len(feature_path_list)-1))
shuffle(feature_path_names)
index = (8 * len(feature_path_names)) // 10

train_feature_path_idx = feature_path_names[:index]
val_feature_path_idx = feature_path_names[index:]

attr_classifier = torch.load('output/trained_model_15_0.0001_4999')
attr_classifier.eval()

total_accuracy = 0
softmax = nn.Softmax()
cnt = 0

for i in range(len(val_feature_path_idx)):
    feature_vec, label_vec = train_iters.load_features_labels([feature_path_list[val_feature_path_idx[i]]], id_labels)
    
    for j in range(len(feature_vec)):
        x = Variable(torch.FloatTensor(np.array(feature_vec)[j])).cuda()
        # x = Variable(torch.FloatTensor(np.array(feature_vec)[j]))
        y_pred = attr_classifier(x)
        label_likelihood = softmax(y_pred).squeeze()
        probs, indices = label_likelihood.topk(1)
        total_accuracy = total_accuracy + is_eq(indices.data.cpu().numpy(), label_vec[j])
        del x, y_pred
    cnt = cnt + len(feature_vec)

print("------- accuracy: %f" %(total_accuracy / cnt))
