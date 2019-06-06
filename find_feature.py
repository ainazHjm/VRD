import torch
import numpy as np
from torchvision import models
from torch import nn
from torch.autograd import Variable
import preprocess

def find_features(id_bbox_dict, id_labels, path_list, model = models.vgg16(pretrained = True)):
    model.cuda().eval()
    classifier = nn.Sequential(*list(model.classifier.children())[0:3]).cuda()
    classifier.eval()
    
    for i in range(len(path_list)):
        bbox_dict = {}      
        im_id = str(path_list[i])[-20:-4]
        
        coords = id_bbox_dict.get(im_id)
        if coords == None:
            continue

        bbox_dict = preprocess.get_bbox(im_id, coords)
        if len(bbox_dict) == 0:
            continue
        # data_feature = []
        for j in range(len(bbox_dict[im_id])):
            data = []
            x = Variable(torch.FloatTensor(bbox_dict[im_id][j])).cuda().view(1, 3, 224, 224)
            
            y = classifier(model.features(x).view(1, -1))
            feature = y.data[0].cpu().numpy()
            norm = np.linalg.norm(feature, 2)
            data.append(feature/norm)

            label = np.zeros(5)
            label[id_labels[im_id][j]] = 1
            data.append(label)
            # print(data)
            np.save(open('bbox_features/'+str(im_id)+'_'+str(j)+'.npy', 'wb+'), np.asarray(data))

            del x, y

        # x = Variable(torch.FloatTensor(np.asarray(bbox_dict[im_id]))).cuda().view(-1, 3, 224, 224)
        # y = classifier(feature_out.view(len(bbox_dict[im_id]), -1))
        
        # np.save(open('features/'+'bboxes'+str(im_id)+'.npy', 'wb+'), y.data.cpu().numpy()) 
        # np.save(open('features/'+'bboxes'+str(im_id)+'.npy', 'wb+'), np.asarray(data_feature)) 
        # saving in the shape of (number of bounding boxes, features) in the same order of bbox_dict

        if (i+1) % 200 == 0:
            print('saved bboxes for %d images' %(i+1))
        # del bbox_dict, x, y, feature_out 
    

        
        