from torch import nn

# Define network model
class xorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        # Define one hidden Layer with Relu activation and one output layer
        self.xor_stack = nn.Sequential(
            nn.Linear(2,2),
            nn.Sigmoid(),
            nn.Linear(2,2)
        )
        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.xor_stack(x)
        return logits
