from torch.autograd import Variable
from tensorboardX import SummaryWriter
import model
import torch.nn as nn
import torch.optim
import numpy as np
import torch

writer = SummaryWriter()

def load_features_labels(path_names):
    feature_vec = [] # feature vector is a vector whose element has all the related bboxes for an image,
                     # the name of the image is the same as the image id
    label_vec = []
    for i in range(len(path_names)):
        f = np.load(str(path_names[i])) # f has two vectors, one of size 4096 and one of size 5

        feature_vec.append(f[0].reshape(1, 4096))
        label_vec.append(f[1].reshape(1, 5))
        # id = str(path_names[i])[-20:-4]
        
        # for idx, feature in enumerate(f):
            # norm = np.linalg.norm(feature, 2)
            # feature_vec.append(feature/norm)
#            print(feature_vec[-1])
            # label = np.zeros(5)
            # label[id_labels[id][idx]] = 1 # id_labels[id] returns a list of ids corresponding to bboxes
            # label_vec.append(label)

    return feature_vec, label_vec        

def validate(model, val_feature_path_idx, feature_path_list, id_labels):
    criterion = nn.MultiLabelSoftMarginLoss()
    num_iters = 3
    batch_size = len(val_feature_path_idx)//num_iters
    avg_loss = 0

    for i in range(num_iters):
        start_idx = i * batch_size  % len(val_feature_path_idx)
        if start_idx+batch_size > len(val_feature_path_idx):
            continue

        val_feature_path = []
        for e in val_feature_path_idx[start_idx:start_idx+batch_size]:
            val_feature_path.append(feature_path_list[e])
        
        feature_vec, label_vec = load_features_labels(val_feature_path, id_labels)
        
        val_x = Variable(torch.FloatTensor(np.array(feature_vec))).cuda()
        # val_x = Variable(torch.FloatTensor(np.array(feature_vec)))
        val_y = Variable(torch.FloatTensor(np.array(label_vec))).cuda()
        # val_y = Variable(torch.FloatTensor(np.array(label_vec)))
        val_y_pred = model(val_x)
        loss = criterion(val_y_pred, val_y)
        avg_loss = avg_loss + loss.item()

        #del val_x, val_y, val_y_pred
    
    return avg_loss/num_iters
        
def train_iters(model, train_feature_path, val_feature_path, learning_rate=0.001, batch_size=20, epochs=2):
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
    scheduler = torch.optim.ReduceLROnPlateau(optimizer, 'min')

    train_losses = []
    val_losses = []
    num_iter = len(train_feature_path_idx)//batch_size
    #print(len(train_feature_path_idx))
    #print(num_iter)
    avg_loss = 0
    train_feature_vec, train_label_vec = load_features_labels(train_feature_path)
    val_feature_vec, val_label_vec = load_features_labels(val_feature_path)

    for i in range(epochs):
        for j in range(num_iter):
            # start_idx = j * batch_size % len(train_feature_path_idx)
            # if start_idx+batch_size > len(train_feature_path_idx):
            #     continue
            
            # train_feature_path = []
            # for idx in train_feature_path_idx[start_idx:start_idx+batch_size]:
                # train_feature_path.append(feature_path_list[idx])

            # feature_vec, label_vec = load_features_labels(train_feature_path, id_labels)
            x = Variable(torch.FloatTensor(np.array(train_feature_vec[j*batch_size:(j+1)*batch_size]))).cuda()
            # x = Variable(torch.FloatTensor(np.array(feature_vec)))
            y = Variable(torch.FloatTensor(np.array(train_label_vec[j*batch_size:(j+1)*batch_size]))).cuda()
            # y = Variable(torch.FloatTensor(np.array(label_vec)))
            
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            
            # writer.add_scalar('loss/train_every_iter', loss.data[0], i)
            # loss.backward()
            # optimizer.step()
            # avg_loss += float(loss)
            
            if (i+1) % 1000 == 0:
                train_losses.append(avg_loss / 1000)
                avg_loss = 0
                val_losses.append(validate(model, val_feature_path_idx, feature_path_list, id_labels))
                print(i, train_losses[-1], val_losses[-1])
                writer.add_scalar('loss/train', train_losses[-1], i)
                writer.add_scalar('loss/validate', val_losses[-1], i)
            
            loss.backward()
            optimizer.step()
            avg_loss += float(loss)

            if (i+1) % 5000 == 0:
                torch.save(model, 'output/trained_model_'+str(epochs)+'_'+str(learning_rate)+'_'+str(i))


            del x, y, y_pred
    
    torch.save(model, 'output/trained_model_'+str(epochs)+'_'+str(learning_rate))
    return train_losses, val_losses    
