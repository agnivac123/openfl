import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import math
import torchvision
from torchvision import transforms
import numpy as np

# Check if GPU (CUDA) is available, else use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========== Data Loading (using your code) ==========
mnist_train = torchvision.datasets.MNIST(
    "./files/",
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]),
)

mnist_test = torchvision.datasets.MNIST(
    "./files/",
    train=False,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]),
)

# ========== Configuration ==========
batch_size_train = 16
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10


random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)



class Sketch():
   
    @staticmethod
    def rand_hashing(n, q, seed=None):
        """Modified to support fractional q ≥ 1.0
        Args:
            n: (integer) number of items to be hashed
            q: (float ≥ 1.0) compression ratio (1.0 = no compression)
        Returns:
            hash_idx: (q_eff-by-s Torch Tensor) where q_eff = floor(q)
            rand_sgn: (n-by-1 Torch Tensor) random ±1 signs
        """
        assert q >= 1.0, "q must be ≥ 1.0"
        s = int(n / q)  # Target sketch size (truncated to integer)
        s = max(1, min(s, n))  # Clamp to valid range [1, n]
        q_eff = math.floor(q) if q >= 2.0 else 1  # Effective q for reshaping

        # Save original RNG state and set seed if provided
        original_rng_state = torch.random.get_rng_state()
        if seed is not None:
            torch.manual_seed(seed)

        t = torch.randperm(n)
        # Handle cases where s*q_eff might exceed n
        max_possible = min(s * q_eff, n)
        hash_idx = t[:max_possible].reshape((q_eff, -1))  # Flexible reshaping
        rand_sgn = torch.randint(0, 2, (n,)).float() * 2 - 1

        # Restore original RNG state
        if seed is not None:
            torch.random.set_rng_state(original_rng_state)
       
        return hash_idx.to(device), rand_sgn.to(device)
   
    @staticmethod
    def countsketch(a, hash_idx, rand_sgn):
        """Unchanged from original"""
        m, n = a.shape
        s = hash_idx.shape[1]
        b = a.mul(rand_sgn)
        c = torch.sum(b[:, hash_idx], dim=1)
        return c
   
    @staticmethod
    def transpose_countsketch(c, hash_idx, rand_sgn):
        """Modified to handle variable q"""
        m, s = c.shape
        n = len(rand_sgn)
        q_eff = hash_idx.shape[0]  # Get q from hash_idx shape
       
        b = torch.zeros([m, n], dtype=torch.float32).to(device)
        if q_eff > 1:
            q = n // s  # Infer original q
            idx = torch.repeat_interleave(torch.arange(s), q, dim=-1)
            selected = hash_idx.T.reshape((-1,))
            b[:, selected] = c[:, idx]
        else:
            b[:, hash_idx[0]] = c  # Special case when q_eff=1
           
        b = b.mul(rand_sgn)
        return b
    
    
class SketchLinearFunction(torch.autograd.Function):
    """
    Modified to ensure all operations stay in sketched space
    Never reconstructs full weights during forward/backward
    """
    @staticmethod
    def forward(ctx, input, sketched_weight, bias, hash_idx, rand_sgn, training=True):
        if training:
            # Sketch input and compute output directly in sketched space
            input_sketch = Sketch.countsketch(input, hash_idx, rand_sgn)
            output = input_sketch @ sketched_weight.t() + bias
            
            # Store only what's needed for backward (all in sketched space)
            ctx.save_for_backward(input_sketch, sketched_weight, bias)
            ctx.hash_idx = hash_idx
            ctx.rand_sgn = rand_sgn
        else:
            # For inference (not used in FL)
            output = input @ Sketch.transpose_countsketch(sketched_weight, hash_idx, rand_sgn).t() + bias
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Compute gradients while maintaining sketching
        Returns:
            grad_input: sketched input gradient
            grad_weight: sketched weight gradient  
            grad_bias: normal bias gradient
        """
        input_sketch, sketched_weight, bias = ctx.saved_tensors
        hash_idx = ctx.hash_idx
        rand_sgn = ctx.rand_sgn
        
        # Compute gradients in sketched space
        grad_sketched_weight = grad_output.t() @ input_sketch
        grad_bias = grad_output.sum(0)
        
        # Sketch the input gradient
        grad_input_sketch = grad_output @ sketched_weight
        grad_input = Sketch.transpose_countsketch(grad_input_sketch, hash_idx, rand_sgn)
        
        return grad_input, grad_sketched_weight, grad_bias, None, None, None


class SketchLinear(nn.Module):
    """
    Layer that ONLY stores and operates on sketched weights
    Never contains full weights at any point
    """
    def __init__(self, input_features, output_features, q=2):
        super(SketchLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.q = q
        
        # Initialize bias normally (small, so no sketching needed)
        self.bias = nn.Parameter(torch.Tensor(output_features))
        
        # Will be initialized by federated flow
        self.sketched_weight = None  
        self.hash_idx = None
        self.rand_sgn = None
        
        # Initialize bias
        bound = 1 / math.sqrt(input_features)
        self.bias.data.uniform_(-bound, bound)

    def init_sketched_weights(self, hash_idx, rand_sgn):
        """Initialize with properly sized random sketched weights"""
        self.hash_idx = hash_idx
        self.rand_sgn = rand_sgn
        sketch_dim = hash_idx.shape[1]  # Get target sketch size
        self.sketched_weight = nn.Parameter(
            torch.randn(self.output_features, sketch_dim) * 0.01
        )

    def forward(self, input):
        # During training: always use sketched operations
        if self.training:
            return SketchLinearFunction.apply(
                input, self.sketched_weight, self.bias, 
                self.hash_idx, self.rand_sgn, self.training
            )
        # During inference: approximate full weights
        else:
            approx_weight = Sketch.transpose_countsketch(
                self.sketched_weight, self.hash_idx, self.rand_sgn
            )
            return F.linear(input, approx_weight, self.bias)


# Multilayer perceptron with sketch
# Args:
    #    dim_in: input dimension
    #    dim_out: output dimension
    #    p: parameter for random hashing in Sketch
# Return:
    #    log probabilities of the classes
class MLP_SketchLinear(nn.Module):
    """
    Modified to properly handle sketched initialization
    and maintain double-blind properties
    """
    def __init__(self, dim_in, dim_out, p):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.p = p
        self.hidden = [200, 200]  # Hidden layer sizes

        # Initialize layers (but don't init weights yet)
        self.input_layer = SketchLinear(dim_in, self.hidden[0], p)
        self.hidden_layers = nn.ModuleList([
            SketchLinear(self.hidden[i], self.hidden[i+1], p) 
            for i in range(len(self.hidden)-1)
        ])
        self.output_layer = nn.Linear(self.hidden[-1], dim_out)  # Final layer doesn't need sketching
        
        self.activation = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def init_sketched_weights(self, hash_idxs, rand_sgns):
        """Initialize all layers with sketched weights"""
        self.input_layer.init_sketched_weights(hash_idxs[0], rand_sgns[0])
        for i, layer in enumerate(self.hidden_layers, start=1):
            layer.init_sketched_weights(hash_idxs[i], rand_sgns[i])
        # Output layer uses normal initialization
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        x = x.view(-1, self.dim_in)
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return self.log_softmax(self.output_layer(x))
    
# Multilayer perceptron
# Args:
    #    dim_in: input dimension
    #    dim_out: output dimension
# Return:
    #    log probabilities of the classes
class NN(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.hidden = [1000, 1000]

        self.input_layer = nn.Linear(self.dim_in, self.hidden[0])
        self.hidden_layers = nn.ModuleList([nn.Linear(self.hidden[i], self.hidden[i + 1])
                                            for i in range(0, len(self.hidden) - 1)])

        self.output_layer = nn.Linear(self.hidden[-1], self.dim_out)
        self.activation = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(-1, self.dim_in)
        output = self.input_layer(x)
        output = self.activation(output)
        for hidden_layers in self.hidden_layers:
            output = hidden_layers(output)
        output = self.activation(output)
        output = self.output_layer(output)
        output = self.log_softmax(output)
        return output
    

def inference(network, test_loader):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = network(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return float(correct) / len(test_loader.dataset)

def FedAvg(models):
    """Generic FedAvg that averages all parameters in the state_dict"""
    new_model = deepcopy(models[0])
    state_dicts = [model.state_dict() for model in models]
    avg_state_dict = {}
    for key in state_dicts[0]:
        avg_state_dict[key] = torch.mean(torch.stack([sd[key] for sd in state_dicts]), dim=0)
    new_model.load_state_dict(avg_state_dict)
    return new_model

from copy import deepcopy

from openfl.experimental.workflow.interface import FLSpec, Aggregator, Collaborator
from openfl.experimental.workflow.runtime import LocalRuntime
from openfl.experimental.workflow.placement import aggregator, collaborator

class FederatedFlow(FLSpec):

    def __init__(self, model=None, optimizer=None, rounds=3, **kwargs):
        super().__init__(**kwargs)
        if model is not None:
            self.model = model
            self.optimizer = optimizer
        else:
            self.model = MLP_SketchLinear(dim_in=784, dim_out=10, p=2)
            # self.model = NN(dim_in=784, dim_out=10)

            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate,
                                       momentum=momentum)
        self.rounds = rounds
        self.current_round = 0 # Initialize here to avoid AttributeError

        self.hash_idxs = []
        self.rand_sgns = []


    def generate_hash_parameters(self):
        """Generate hashes for all sketch layers"""
        self.hash_idxs = []
        self.rand_sgns = []
        
        # Generate for input layer (784->1000)
        h, r = Sketch.rand_hashing(784, self.model.p, seed=42+self.current_round)
        self.hash_idxs.append(h)
        self.rand_sgns.append(r)
        
        # Generate for hidden layers (1000->1000)
        for _ in range(len(self.model.hidden_layers)):
            h, r = Sketch.rand_hashing(200, self.model.p, seed=43+self.current_round)
            self.hash_idxs.append(h)
            self.rand_sgns.append(r)

    
    
    @aggregator
    def start(self):
        print(f'Initializing hash parameters and model..')
        self.generate_hash_parameters()  # Generate first set of hash parameters
        self.collaborators = self.runtime.collaborators
        self.private = 10
        # self.current_round = 0

        # Initialize model with sketches
        self.model.init_sketched_weights(self.hash_idxs, self.rand_sgns)

        self.next(self.aggregated_model_validation,
                  foreach='collaborators',
                  hash_idxs=self.hash_idxs,  # Broadcast hash parameters for next round
                  rand_sgns=self.rand_sgns,  # Broadcast hash parameters for next round
                  exclude=['private'])

    
    @collaborator
    def aggregated_model_validation(self):
        print(f'Performing aggregated model validation for collaborator {self.input}')

        # Set hash parameters from the received state
        self.model.hash_idxs = self.hash_idxs
        self.model.rand_sgns = self.rand_sgns

        self.agg_validation_score = inference(self.model, self.test_loader)
        print(f'{self.input} value of {self.agg_validation_score}')
        self.next(self.train)
    
    @collaborator
    def train(self):
        self.model.train()
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate,
                                   momentum=momentum)
        train_losses = []
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: 1 [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.item()))
                self.loss = loss.item()
                torch.save(self.model.state_dict(), 'model.pth')
                torch.save(self.optimizer.state_dict(), 'optimizer.pth')
        self.training_completed = True
        self.next(self.local_model_validation)

    @collaborator
    def local_model_validation(self):
        self.local_validation_score = inference(self.model, self.test_loader)
        print(
            f'Doing local model validation for collaborator {self.input}: {self.local_validation_score}')
        self.next(self.join, exclude=['training_completed'])
    

    @aggregator
    def join(self, inputs):
        self.average_loss = sum(input.loss for input in inputs) / len(inputs)
        self.aggregated_model_accuracy = sum(
            input.agg_validation_score for input in inputs) / len(inputs)
        self.local_model_accuracy = sum(
            input.local_validation_score for input in inputs) / len(inputs)
        print(f'Average aggregated model validation values = {self.aggregated_model_accuracy}')
        print(f'Average training loss = {self.average_loss}')
        print(f'Average local model validation values = {self.local_model_accuracy}')
        
        # Use the generic FedAvg
        self.model = FedAvg([input.model for input in inputs])
        
        self.optimizer = [input.optimizer for input in inputs][0]
        self.current_round += 1
        if self.current_round < self.rounds:
            # Generate new hash parameters for the next round
            self.generate_hash_parameters()
            
            self.next(self.aggregated_model_validation,
                    foreach='collaborators',
                    hash_idxs=self.hash_idxs,  # Broadcast new hash parameters
                    rand_sgns=self.rand_sgns,
                    exclude=['private'])
        else:
            self.next(self.end)

    @aggregator
    def end(self, *args, **kwargs):  # Fix signature
        print(f'This is the end of the flow')


import time
start_time = time.time()

# Setup participants
aggregator = Aggregator()
aggregator.private_attributes = {}

# Setup collaborators with private attributes
collaborator_names = ['Portland', 'Seattle', 'Chandler','Bangalore']
collaborators = [Collaborator(name=name) for name in collaborator_names]
for idx, collaborator in enumerate(collaborators):
    local_train = deepcopy(mnist_train)
    local_test = deepcopy(mnist_test)
    local_train.data = mnist_train.data[idx::len(collaborators)]
    local_train.targets = mnist_train.targets[idx::len(collaborators)]
    local_test.data = mnist_test.data[idx::len(collaborators)]
    local_test.targets = mnist_test.targets[idx::len(collaborators)]
    collaborator.private_attributes = {
            'train_loader': torch.utils.data.DataLoader(local_train,batch_size=batch_size_train, shuffle=True),
            'test_loader': torch.utils.data.DataLoader(local_test,batch_size=batch_size_train, shuffle=True)
    }

local_runtime = LocalRuntime(aggregator=aggregator, collaborators=collaborators, backend='single_process')
print(f'Local runtime collaborators = {local_runtime.collaborators}')

model = None
best_model = None
optimizer = None
flflow = FederatedFlow(model, optimizer, rounds=3, checkpoint=False)
flflow.runtime = local_runtime
flflow.run()

# After the flow completes, calculate and print the total time
end_time = time.time()
total_time = end_time - start_time
print(f"\nTotal time for federated flow: {total_time:.2f} seconds")