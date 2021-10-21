import torch
import logging
import torch.nn.functional as F
logger = logging.getLogger(__name__)

# Computes Loss for REINFORCE 
class Reinforce_Loss(torch.nn.NLLLoss):
    def forward(self, input, target, G, T):
        nll_result = F.nll_loss(input, target, reduction='none')
        return torch.div(torch.dot(G, nll_result), T)

# TODO Consider ignoring output_activation, and using L = torch.nn.CrossEntropyLoss
#
class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, output_size, activation, layers=[64, 64, 64, 32, 32, 16]):
        super().__init__()

        self.linear1 = torch.nn.Linear(input_size, layers[0])
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(layers[0], layers[1])
        self.activation2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(layers[1], layers[2])
        self.activation3 = torch.nn.ReLU()
        self.linear4 = torch.nn.Linear(layers[2], layers[3])
        self.activation4 = torch.nn.ReLU()
        self.linear5 = torch.nn.Linear(layers[3], layers[4])
        self.activation5 = torch.nn.ReLU()
        self.linear6 = torch.nn.Linear(layers[4], layers[5])
        self.activation6 = torch.nn.ReLU()

        self.output_layer = torch.nn.Linear(layers[5], output_size)
        self.output_activation = activation

        #initialize weights, following 'fan_avg' approach
        torch.nn.init.xavier_normal_(self.linear1.weight)
        torch.nn.init.xavier_normal_(self.linear2.weight)
        torch.nn.init.xavier_normal_(self.linear3.weight)
        torch.nn.init.xavier_normal_(self.output_layer.weight)

    def forward(self, inputs):
        x = self.activation1(self.linear1(inputs))
        x = self.activation2(self.linear2(x))
        x = self.activation3(self.linear3(x))
        x = self.activation4(self.linear4(x))
        x = self.activation5(self.linear5(x))
        x = self.activation6(self.linear6(x))
        x = self.output_activation(self.output_layer(x))
        # logger.debug("output of NN ==> %s", str(x))
        # logger.debug("output of argmax(x) ==> %s", str(torch.argmax(x)))
        return x
