import torch.nn as nn
import torch.nn.functional as F

class MultiLabelClassifier(nn.Module):
    def __init__(self, hidden_size1, hidden_size2, output_size):
        super(MultiLabelClassifier, self).__init__()
        self.feature_size = 4096 # VGG16  classifier output size
        self.net = nn.Sequential(
            nn.Linear(self.feature_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, output_size),
#            nn.Softmax(),
	)
    
    def forward(self, inp_feature):
        return self.net(inp_feature)
