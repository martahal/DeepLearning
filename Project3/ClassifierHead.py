from torch import nn


class ClassifierHead(nn.Module):

    def __init__(self,
                 input_size,
                 output_size):

        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.model = nn.Sequential(
            nn.Linear(in_features=input_size,out_features=output_size)
        )
