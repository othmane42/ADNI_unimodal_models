import torch.nn as nn
import torch
class CustomClassifier(nn.Module):
    def __init__(self, hidden_dim, activation_fun, num_class,task, dropout_rate=None) -> None:
        super(CustomClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.activation_layer1 = activation_fun
        self.num_class = num_class
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate else None
        self.task = task

        # Lazy layer initialization
        self.fc1 = None  # Placeholder for the first layer
        self.fc2 = None  # Placeholder for the second layer
        self.in_dim_initialized = False  # Track whether in_dim has been set

    def set_input_dim(self, in_dim,device):
        """
        Initialize layers with the given input dimension.
        This should only be called once.
        """
        if not self.in_dim_initialized:
            print("Initializing layers with input dimension:", in_dim)
            if self.hidden_dim != 0:
                self.fc1 = nn.Linear(in_dim, self.hidden_dim).to(device)
                self.fc2 = nn.Linear(self.hidden_dim, 1 if self.task=="binary" else self.num_class).to(device)
            else:
                self.fc2 = nn.Linear(in_dim, 1 if self.task=="binary" else self.num_class).to(device)
            self.in_dim_initialized = True

    def forward(self, x):
        # One-time initialization of layers
        if not self.in_dim_initialized:
            self.set_input_dim(x.shape[-1],x.device)

        # Forward pass
        if self.hidden_dim != 0:
            x = self.fc1(x)
            x = self.activation_layer1(x)
            if self.dropout:
                x = self.dropout(x)
        x = self.fc2(x)
        #x = x.squeeze(dim=-1)
        return x