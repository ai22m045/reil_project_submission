import torch
import torch.nn as nn
import os

class Net(nn.Module):
    def __init__(self, board_size: int, hidden=0, hidden_width=343, num_filters=49, kernel_size=3):
        super(Net, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv2d(1, num_filters, kernel_size)
        self.tanh = nn.Tanh()

        # Flatten Layer
        self.flatten = nn.Flatten(start_dim=1)

        # MLP Layers
        self.mlp_layers = nn.Sequential()
        input_size = num_filters * (((board_size - (kernel_size - 1))) ** 2)  # Calculate input size for MLP
        self.y = input_size

        if hidden > 0:
            self.mlp_layers.add_module("linear_0", nn.Linear(input_size, hidden_width))
            self.mlp_layers.add_module("relu_0", nn.ReLU())
            for i in range(1, hidden + 1):
                self.mlp_layers.add_module("linear_" + str(i), nn.Linear(hidden_width, hidden_width))
                self.mlp_layers.add_module("relu_" + str(i), nn.ReLU())

        # Output Layer
        self.output_layer = nn.Linear(hidden_width, board_size ** 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.tanh(x)
        x = self.flatten(x)
        x = self.mlp_layers(x)
        x = self.output_layer(x)
        return x

def scalar_to_coordinates (scalar):
    """
    Helper function to transform a scalar "back" to coordinates.
    Reverses the output of 'coordinate_to_scalar'.
    """
    coord1 = int(scalar/7)
    coord2 = scalar - coord1 * 7
    assert(0 <= coord1 and 6 >= coord1), "The scalar input is invalid."
    assert(0 <= coord2 and 6 >= coord2), "The scalar input is invalid."
    return (coord1, coord2)

player_1 = Net(board_size=7, hidden=3, hidden_width=7**4)
player_1.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),"hex7_1_white.pt")))

player_2 = Net(board_size=7, hidden=3, hidden_width=7**4)
player_2.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),"hex7_1_black.pt")))

#stupid example
def agent (board, action_set):
    state = torch.tensor([board], dtype=torch.float32).unsqueeze(0)
    legal_moves_mask = (state == 0)
    #player 1's turn
    if state.sum() == 0:
        with torch.no_grad():
            # for argmax
            output = player_1(state)
            # print(output)
            # print(torch.softmax(output/output.max(), dim=1))
            legal_output = output.masked_fill(~legal_moves_mask.flatten(), float('-inf'))
            action =  legal_output.argmax().unsqueeze(0)
    #player 2's turn
    else:
        with torch.no_grad():
            # for argmax
            output = player_2(state)
            # print(output)
            # print(torch.softmax(output/output.max(), dim=1))
            legal_output = output.masked_fill(~legal_moves_mask.flatten(), float('-inf'))
            action =  legal_output.argmax().unsqueeze(0)
    #change action into coordinates
    output = scalar_to_coordinates(action.item())
    return output

#Here should be the necessary Python wrapper for your model, in the form of a callable agent, such as above.
#Please make sure that the agent does actually work with the provided Hex module.