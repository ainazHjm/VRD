from torch.autograd import Variable
from tensorboardX import SummaryWriter
import model
import torch.nn as nn
import torch.optim
import numpy as np
import torch
import random

writer = SummaryWriter()

def load_features_labels(path_names):
    feature_vec = []
    label_vec = []
    for i in range(len(path_names)):
        f = np.load(str(path_names[i])) # f has two vectors, one of size 4096 and one of size 5

        feature_vec.append(f[0].reshape(1, 4096))
        label_vec.append(f[1].reshape(1, 5))
        # id = str(path_names[i])[-20:-4]
    return feature_vec, label_vec

def validate(model, val_feature_vec, val_label_vec):
    criterion = nn.MultiLabelSoftMarginLoss()
    # num_iters = 3
    # batch_size = len(val_feature_vec)//num_iters
    # avg_loss = 0

    # for i in range(num_iters):
    val_x = Variable(torch.FloatTensor(np.array(val_feature_vec))).cuda()
    val_y = Variable(torch.FloatTensor(np.array(val_label_vec))).cuda()
    val_y_pred = model(val_x)
    loss = criterion(val_y_pred, val_y)
    # avg_loss += float(loss)
        # del val_x, val_y, val_y_pred
    return float(loss)

def is_eq(pred_indices, label_vec):
    for idx in pred_indices:
        if label_vec[idx] == 1:
            return 1
    return 0

def find_accuracy(model, feature_vec, label_vec):
    total_accuracy = 0
    softmax = nn.Softmax()
    for i in range(len(feature_vec)):
        x = Variable(torch.FloatTensor(feature_vec[i])).cuda()
        y_pred = model(x)
        label_likelihood = softmax(y_pred).squeeze()
        probs, indices = label_likelihood.topk(1)
        total_accuracy = total_accuracy + is_eq(indices, label_vec[i][0])

    return total_accuracy/len(feature_vec)

def train_iters(model, train_feature_path, val_feature_path, learning_rate=0.01, batch_size=20, epochs=50):
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    train_loss, val_loss, avg_loss = 0, 0, 0
    train_accuracy, val_accuracy = 0, 0
    num_iter = len(train_feature_path)//batch_size

    train_feature_vec, train_label_vec = load_features_labels(train_feature_path)
    val_feature_vec, val_label_vec = load_features_labels(val_feature_path)
    print('*** loaded train and val features')

    for i in range(epochs):
        for j in range(num_iter):
            x = Variable(torch.FloatTensor(np.array(train_feature_vec[j*batch_size:(j+1)*batch_size]))).cuda()
            y = Variable(torch.FloatTensor(np.array(train_label_vec[j*batch_size:(j+1)*batch_size]))).cuda()

            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            avg_loss += float(loss)

            loss.backward()
            optimizer.step()

            del x, y, y_pred

            if (j+1) % 1000 == 0:
                train_loss = avg_loss / 1000
                avg_loss = 0
                val_loss = validate(model, val_feature_vec, val_label_vec)
                scheduler.step(val_loss)
                print('epoch %d: %d --- %f, %f' %(i, j, train_loss, val_loss))
                writer.add_scalar('loss/train', train_loss, i)
                writer.add_scalar('loss/validate', val_loss, i)

            if (j+1) % 5000 == 0:
                torch.save(model, 'output/trained_model_'+str(epochs)+'_'+str(learning_rate)+'_'+str(j))

        train_accuracy = find_accuracy(model, train_feature_vec, train_label_vec)
        val_accuracy = find_accuracy(model, val_feature_vec, val_label_vec)
        print('----- validation accuracy: %f, train_accuracy: %f' %(val_accuracy, train_accuracy))
        writer.add_scalar('accuracy/train', train_accuracy)
        writer.add_scalar('accuracy/val', val_accuracy)

    torch.save(model, 'output/final_model_'+str(epochs)+'_'+str(learning_rate))
    return train_accuracy, val_accuracy
