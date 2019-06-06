import torch.nn as nn
import torch.nn.functional as F

class MultiLabelClassifier(nn.Module):
    def __init__(self, hidden_size1, hidden_size2, output_size):
        super(MultiLabelClassifier, self).__init__()
        self.feature_size = 4096 # VGG16  classifier output size
        # self.feature = feature_model.features
        # self.classifier = nn.Sequential(*list(feature_model.classifier.children())[:-1])
        self.fc1 = nn.Linear(self.feature_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
    
    def forward(self, inp_feature):
        # f = self.feature(inp)
        # f = f.view(f.size(0), -1)
        # f = self.classifier(inp_feature)
        out = self.fc3(F.sigmoid(self.fc2(F.sigmoid(self.fc1(inp_feature)))))
        return out
