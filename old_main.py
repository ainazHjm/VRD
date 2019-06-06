import preprocess
import find_feature
import model
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

p = Path('/home/mehran/Desktop/workspace/get images/images/')
# p = Path('../data/sampled/')
q = list(p.glob('*.jpg'))

img_id, id_bbox_dict, id_labels = preprocess.prepare_data(attr_lables)
print("1. prepared data - check")

find_feature.find_features(id_bbox_dict, id_labels, q) # saving features for all images (with 'is' relation and not corrupt)
print('2. saved features - check')
# print('features are already saved - check')

feature_path = Path('bbox_features/')
feature_path_list = list(feature_path.glob('*.npy'))

feature_path_names = list(range(len(feature_path_list)))
shuffle(feature_path_names)
index = (9 * len(feature_path_names)) // 10

# train_feature_path_idx = feature_path_names[:index]
# val_feature_path_idx = feature_path_names[index:]

train_feature_path = []
for idx in feature_path_names[:index]:
    train_feature_path.append(feature_path_list[idx])

val_feature_path = []
for idx in feature_path_names[index:]:
    val_feature_path.append(feature_path_list[idx])

print(len(train_feature_path))
print(len(val_feature_path))

print('3. separated train and validation - check')

attr_classifier = model.MultiLabelClassifier(300, 40, 5).cuda()
# attr_classifier = model.MultiLabelClassifier(300, 40, 5)
attr_classifier.eval()
print("4. model created - check")

train_iters.train_iters(attr_classifier, train_feature_path, val_feature_path)
# print("5. train completed - check")

# total_accuracy = 0
# softmax = nn.Softmax()
# cnt = 0

# for i in range(len(val_feature_path_idx)):
#     feature_vec, label_vec = train_iters.load_features_labels([feature_path_list[val_feature_path_idx[i]]], id_labels)
    
#     for j in range(len(feature_vec)):
#         x = Variable(torch.FloatTensor(np.array(feature_vec)[j])).cuda()
#         # x = Variable(torch.FloatTensor(np.array(feature_vec)[j]))
#         y_pred = attr_classifier(x)
#         label_likelihood = softmax(y_pred).squeeze()
#         probs, indices = label_likelihood.topk(1)
#         total_accuracy = total_accuracy + is_eq(indices, label_vec[j])
#         del x, y_pred
#     cnt = cnt + len(feature_vec)

# print("------- accuracy: %f" %(total_accuracy / cnt))
