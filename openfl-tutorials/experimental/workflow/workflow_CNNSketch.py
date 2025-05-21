#============================
#######  CONFIGURATION ######
#============================
USE_MNIST    = False     # set True to train on MNIST, False --> CIFAR10
USE_SKETCH   = False     # set True to use the "_Sketch" versions of the models
USE_SEP      = True     # set True for depthwise-separable convolution


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision
import numpy as np
import math
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size_train = 128
batch_size_test = 1000
learning_rate = 0.001
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

#==================================
###### DATASETS & TRANSFORMS ######
#==================================
if USE_MNIST:
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_train = torchvision.datasets.MNIST(
        "./files", train=True,  download=True, transform=transform)
    full_test  = torchvision.datasets.MNIST(
        "./files", train=False, download=True, transform=transform)
else:
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616),
        )
    ])
    full_train = torchvision.datasets.CIFAR10(
        "./files", train=True,  download=True, transform=transform)
    full_test  = torchvision.datasets.CIFAR10(
        "./files", train=False, download=True, transform=transform)


#===========================================================
##### CLASSES FOR GENERATING AND APPLYING THE SKETCHES #####
#===========================================================

class Sketch():
   
    @staticmethod
    def rand_hashing(n, q):
        """
        Generate a (possibly identity) hashing scheme for CountSketch.

        Args:
            n (int): Original input dimension.
            q (float): Sketch compression factor (>=1.0). A larger q means more compression.

        Returns:
            hash_idx (LongTensor[q_eff, s]): Indices mapping original features into s buckets,
                                             repeated q_eff times.
            rand_sgn (FloatTensor[n]): Random +-1 signs for each of the n original features.

        Exception: If n <= 5 or q >= n, returns an 'identity' sketch of size n --> n:
          hash_idx = [[0, 1, 2, ..., n-1]]
          rand_sgn = [1, 1, ..., 1]
        Otherwise behaves as before.
        """
        assert q >= 1.0, "q must be >= 1.0"

        # Hard-coded identity for very small dimensions or no compression
        if n <= 5 or q >= n:
            hash_idx = torch.arange(n, device=device).unsqueeze(0).long()  # shape (1, n)
            rand_sgn = torch.ones(n, device=device).float()               # no sign flipping
            return hash_idx, rand_sgn
        # if n <= 5 or q >= n:
        #     perm = torch.randperm(n, device=device)
        #     hash_idx = perm.unsqueeze(0).long()                      # shape (1, n)
        #     rand_sgn = (torch.randint(0, 2, (n,), device=device).float() * 2 - 1)
        #     return hash_idx, rand_sgn

        # Target sketch size for larger dimensions
        s = max(1, min(int(n / q), n))
        # Effective q, clamped so that q_eff * s <= n
        raw_qeff = math.floor(q) if q >= 2.0 else 1
        q_eff = min(raw_qeff, n // s) or 1

        perm = torch.randperm(n, device=device)
        # Now total <= n guaranteed
        total = q_eff * s
        flat = perm[:total]
        hash_idx = flat.reshape(q_eff, -1)
        rand_sgn = (torch.randint(0, 2, (n,), device=device).float() * 2 - 1)
        return hash_idx, rand_sgn
    
    @staticmethod
    def countsketch(a, hash_idx, rand_sgn):
        """
        Apply CountSketch to input matrix a.

        Args:
            a (Tensor[m, n]): Input data with n features per row, m rows.
            hash_idx (LongTensor[q_eff, s]): Hash indices from rand_hashing.
            rand_sgn (FloatTensor[n]): Random +-1 signs for features.

        Returns:
            c (Tensor[m, s]): Sketched output, summing signed features into s buckets.
        """
        m, n = a.shape
        s = hash_idx.shape[1]
        b = a.mul(rand_sgn)
        c = torch.sum(b[:, hash_idx], dim=1)
        return c


class SketchConvChannel(nn.Module):
    """
    A convolutional layer that applies CountSketch to both the input activations
    and the convolutional weights, reducing the channel dimension from in_c --> s_ch.

    Args:
        in_c (int):    Number of input channels.
        out_c (int):   Number of output channels (after convolution).
        kernel_size (int): Size of the 2D convolutional kernel.
        padding (int):     Padding added to all four sides of the input.
        q (float):         Sketch compression factor (>=1.0). Larger q --> more compression.
    """
    def __init__(self, in_c, out_c, kernel_size, padding=0, q=2):
        super().__init__()
        self.in_c, self.out_c = in_c, out_c
        self.k, self.padding, self.q = kernel_size, padding, q

        # weight & bias
        fan_in = in_c * kernel_size * kernel_size
        bound = 1.0/math.sqrt(fan_in)
        self.weight = nn.Parameter(
            torch.empty(out_c, in_c, kernel_size, kernel_size, device=device).uniform_(-bound, bound)
        )

        self.bias = nn.Parameter(
            torch.empty(out_c, device=device).uniform_(-bound, bound)
        )

        # 1) sketch channels of activations (in_c --> s_ch)
        self.hash_idx_ch, self.rand_sgn_ch = Sketch.rand_hashing(in_c, q)
        self.s_ch = self.hash_idx_ch.shape[1]

    def forward(self, x):
        b, C, H, W = x.shape
        # 1) sketch activations per‐pixel: (b,H,W,C) ----> (b,H,W,s_ch)
        x_flat = x.permute(0,2,3,1).reshape(-1, C)                     # (b*H*W, C)
        x_sk_ch= Sketch.countsketch(x_flat, self.hash_idx_ch, self.rand_sgn_ch)  # (b*H*W, s_ch)
        x_sk_ch= x_sk_ch.view(b, H, W, self.s_ch).permute(0,3,1,2)      # (b, s_ch, H, W)

        # 2) sketch weights across channels only:
        #    weight: (out_c, in_c, k, k) ----> (out_c*k*k, in_c)
        w_flat = self.weight.reshape(self.out_c * self.k * self.k, C)
        w_sk   = Sketch.countsketch(w_flat, self.hash_idx_ch, self.rand_sgn_ch)    # (out_c*k*k, s_ch)
        #    reshape back to conv‐kernel: (out_c, s_ch, k, k)
        w_sk4  = w_sk.view(self.out_c, self.k * self.k, self.s_ch) \
                     .permute(0,2,1) \
                     .view(self.out_c, self.s_ch, self.k, self.k)

        # 3) native conv2d on sketched activations
        return F.conv2d(x_sk_ch, w_sk4, bias=self.bias, padding=self.padding)


class SketchLinear(nn.Module):
    """
    A fully-connected (linear) layer that applies CountSketch to both the input features
    and the weight matrix, reducing the feature dimension from in_f --> s_in.

    Args:
        in_f (int):  Number of input features.
        out_f (int): Number of output features.
        q (float):   Sketch compression factor (>=1.0). Larger q --> more compression.
    """
    def __init__(self, in_f, out_f, q):
        super().__init__()
        self.in_f, self.out_f, self.q = in_f, out_f, q
        bound = 1.0/math.sqrt(in_f)
        self.weight = nn.Parameter(
            torch.empty(out_f, in_f, device=device).uniform_(-bound, bound)
        )

        self.bias = nn.Parameter(
            torch.empty(out_f, device=device).uniform_(-bound, bound)
        )

        # sketch input features ----> s_in and
        # sketch weight rows ----> s_wt
        self.hash_idx, self.rand_sgn = Sketch.rand_hashing(in_f, q)

    def forward(self, x):
        # x: (batch, in_f)
        # 1) sketch x's features
        x_sk = Sketch.countsketch(x, self.hash_idx, self.rand_sgn)  # (batch, s_in)
        # 2) sketch each row of weight
        w_sk = Sketch.countsketch(self.weight, self.hash_idx, self.rand_sgn)  # (out_f, s_wt)
        # 3) plain linear
        return F.linear(x_sk, w_sk, self.bias)
    
class SepConv(nn.Module):
    """
    Depthwise-separable convolution block:
      1) Depthwise conv (groups=in_channels)
      2) Pointwise 1x1 conv to mix channels
    This reduces computation vs. a standard conv by splitting spatial and channel mixing.
    """
    def __init__(self, in_c, out_c, k, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_c, in_c, k, padding=padding, groups=in_c)
        self.pointwise = nn.Conv2d(in_c, out_c, 1)
    def forward(self, x):
        return self.pointwise(self.depthwise(x))

#===========================================================    
################### MODEL CLASSES ##########################
#===========================================================
class CNNMnist(nn.Module):
    """
    A simple convolutional network for MNIST digit classification.

    Architecture:
      1) Conv2d --> ReLU --> MaxPool
      2) Conv2d --> ReLU --> MaxPool
      3) Flatten remaining feature maps into a vector
      4) Fully-connected --> ReLU
      5) Final linear --> log-softmax

    Input:
      x: Tensor of shape (batch_size, 1, height, width)

    Output:
      Log-probabilities over 10 classes, shape (batch_size, 10)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 5)
        self.conv2 = nn.Conv2d(64,128, 5)
        self.fc1   = nn.Linear(2048, 512)
        self.fc2   = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2(x),2))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)

class CNNMnist_Sketch(nn.Module):
    """
    A CountSketch-augmented CNN for MNIST.

    Replaces the standard convolution and first linear layers with
    sketch-based versions that randomly project channels/features
    into a lower-dimensional subspace, then applies the same overall
    architecture pattern.

    Steps:
      1) SketchConvChannel --> ReLU --> MaxPool
      2) SketchConvChannel --> ReLU --> MaxPool
      3) Flatten to a vector
      4) SketchLinear --> ReLU
      5) Final linear --> log-softmax

    Args:
      q (float): Sketch compression factor (>=1.0).

    Input:
      x: Tensor of shape (batch_size, 1, height, width)

    Output:
      Log-probabilities over 10 classes, shape (batch_size, 10)
    """
    def __init__(self, q=4):
        super().__init__()
        self.conv1 = SketchConvChannel(1, 64, 5, q=q)
        self.conv2 = SketchConvChannel(64,128, 5, q=q)
        self.fc1   = SketchLinear(2048, 512, q=q)
        self.fc2   = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2(x),2))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)

class CNNCifar(nn.Module):
    """
    A convolutional neural network for CIFAR-10 image classification.

    Architecture:
      1) Conv2d --> BatchNorm --> ReLU
      2) Conv2d --> BatchNorm --> ReLU
      3) MaxPool
      4) Conv2d --> BatchNorm --> ReLU
      5) MaxPool
      6) Flatten remaining feature maps into a vector
      7) Fully-connected --> ReLU
      8) Final linear --> log-softmax

    Input:
      x: Tensor of shape (batch_size, 3, height, width)

    Output:
      Log-probabilities over 10 classes, shape (batch_size, 10)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.BatchNorm2d(32), nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,64, 5, padding=2),
            nn.BatchNorm2d(64), nn.ReLU())
        self.pool  = nn.MaxPool2d(2,2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64,128,5, padding=2),
            nn.BatchNorm2d(128), nn.ReLU())
        self.fc1   = nn.Linear(8192, 200)
        self.fc2   = nn.Linear(200, 10)

    def forward(self, x):
        x = self.pool(self.conv2(self.conv1(x)))
        x = self.pool(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)

class CNNCifar_Sketch(nn.Module):
    """
    A CountSketch-augmented CNN for CIFAR-10.

    Replaces the standard convolution and first linear layers with
    sketch-based versions that randomly project channels/features
    into a lower-dimensional subspace before applying the same
    overall architectural pattern.

    Steps:
      1) SketchConvChannel --> BatchNorm --> ReLU
      2) SketchConvChannel --> BatchNorm --> ReLU
      3) MaxPool
      4) SketchConvChannel --> BatchNorm --> ReLU
      5) MaxPool
      6) Flatten to a vector
      7) SketchLinear --> ReLU
      8) Final linear --> log-softmax

    Args:
      q (float): Sketch compression factor (>=1.0)

    Input:
      x: Tensor of shape (batch_size, 3, height, width)

    Output:
      Log-probabilities over 10 classes, shape (batch_size, 10)
    """
    def __init__(self, q=4):
        super().__init__()
        self.conv1 = nn.Sequential(
            SketchConvChannel(3,32,5, padding=2, q=q),
            nn.BatchNorm2d(32), nn.ReLU())
        self.conv2 = nn.Sequential(
            SketchConvChannel(32,64,5, padding=2, q=q),
            nn.BatchNorm2d(64), nn.ReLU())
        self.pool  = nn.MaxPool2d(2,2)
        self.conv3 = nn.Sequential(
            SketchConvChannel(64,128,5, padding=2, q=q),
            nn.BatchNorm2d(128), nn.ReLU())
        self.fc1   = SketchLinear(8192,200, q=q)
        self.fc2   = nn.Linear(200, 10)

    def forward(self, x):
        x = self.pool(self.conv2(self.conv1(x)))
        x = self.pool(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)
    
class CNNCifar_SepConv(nn.Module):
    """
    A CIFAR-10 CNN using depthwise-separable convolutions.

    Architecture matches CNNCifar but replaces each Conv2d->BN->ReLU block with:
      SepConv(in_c->out_c) -> BatchNorm -> ReLU -> MaxPool

    This design (MobileNet-style) reduces FLOPs by
    decoupling spatial (depthwise) and channel (pointwise) operations.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.BatchNorm2d(32), nn.ReLU())
        self.conv2 = nn.Sequential(
            SepConv(32,64, 5, padding=2),
            nn.BatchNorm2d(64), nn.ReLU())
        self.pool  = nn.MaxPool2d(2,2)
        self.conv3 = nn.Sequential(
            SepConv(64,128,5, padding=2),
            nn.BatchNorm2d(128), nn.ReLU())
        self.fc1   = nn.Linear(8192, 200)
        self.fc2   = nn.Linear(200, 10)

    def forward(self, x):
        x = self.pool(self.conv2(self.conv1(x)))
        x = self.pool(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)
    

    
def inference(network,test_loader):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
      for data, target in test_loader:
        output = network(data)
        test_loss += F.nll_loss(output, target, size_average=False).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
      test_loss, correct, len(test_loader.dataset),
      100. * correct / len(test_loader.dataset)))
    accuracy = float(correct / len(test_loader.dataset))
    return accuracy


from copy import deepcopy
from openfl.experimental.workflow.interface import FLSpec, Aggregator, Collaborator
from openfl.experimental.workflow.runtime import LocalRuntime
from openfl.experimental.workflow.placement import aggregator, collaborator

def FedAvg(models, weights=None):
    new_model = models[0]
    state_dicts = [model.state_dict() for model in models]
    state_dict = new_model.state_dict()
    for key in models[1].state_dict():        
        arr = np.array([state[key].numpy() for state in state_dicts])
        avg = np.average(arr, axis=0, weights=weights)
        state_dict[key] = torch.from_numpy(np.array(avg))
    new_model.load_state_dict(state_dict)
    return new_model



class FederatedFlow(FLSpec):

    def __init__(self, model=None, optimizer=None, rounds=3, **kwargs):
        super().__init__(**kwargs)
        
        if model is not None:
            self.model = model
            self.optimizer = optimizer
        else:
            # choose architecture by dataset + sketch flag
            if USE_MNIST:
                if USE_SKETCH:
                    self.model = CNNMnist_Sketch(q=8).to(device)
                else:
                    self.model = CNNMnist().to(device)
            
            else:
                if USE_SEP:
                    self.model = CNNCifar_SepConv().to(device)
                elif USE_SKETCH: 
                    self.model = CNNCifar_Sketch(q=8).to(device)
                else:
                    self.model = CNNCifar().to(device)
                
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate, momentum=momentum
            )
        
        self.rounds = rounds

        # for timing
        self.cumulative_time = 0.0
        # mark the very start (so round 1 time = now − this)
        self._last_timestamp = time.time()

    @aggregator
    def start(self):
        print(f'Performing initialization for model')

        # --- clear any old logs at the beginning of this run ---
        # determine filenames
        model_name  = self.model.__class__.__name__
        metric_acc  = "aggregated_model_accuracy"
        metric_time = "cumulative_training_time"
        metric_loss = "average_training_loss"

        if "Sketch" in model_name:
            try:
                q = self.model.conv1[0].q
            except Exception:
                q = self.model.conv1.q
            q_str = f"{q:g}".replace('.', 'p')
            fname_acc  = f"{model_name}_q{q_str}_{metric_acc}.txt"
            fname_time = f"{model_name}_q{q_str}_{metric_time}.txt"
            fname_loss = f"{model_name}_q{q_str}_{metric_loss}.txt"
        else:
            fname_acc  = f"{model_name}_{metric_acc}.txt"
            fname_time = f"{model_name}_{metric_time}.txt"
            fname_loss = f"{model_name}_{metric_loss}.txt"

        # truncate all three logs
        open(fname_acc,  "w").close()
        open(fname_time, "w").close()
        open(fname_loss, "w").close()
        # ---------------------------------------------------------

        self.collaborators = self.runtime.collaborators
        self.private = 10
        self.current_round = 0
        self.next(self.aggregated_model_validation, foreach='collaborators', exclude=['private'])

    @collaborator
    def aggregated_model_validation(self):
        print(f'Performing aggregated model validation for collaborator {self.input}')
        self.agg_validation_score = inference(self.model, self.test_loader)
        print(f'{self.input} value of {self.agg_validation_score}')
        self.next(self.train)

    @collaborator
    def train(self):

        # print collaborator name and which round we’re in
        print(f'Collaborator {self.input} rounds [{self.current_round+1}/{self.rounds}]')
        epoch_start = time.time()

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
                print(f"Train Epoch: [{batch_idx*len(data):5d}/"
                      f"{len(self.train_loader.dataset):5d} "
                      f"({100.*batch_idx/len(self.train_loader):3.0f}%)]\t"
                      f"Loss: {loss.item():.6f}"
                      )
                
                self.loss = loss.item()
                torch.save(self.model.state_dict(), 'model.pth')
                torch.save(self.optimizer.state_dict(), 'optimizer.pth')

        epoch_time = time.time() - epoch_start

        print(f'Collaborator {self.input} finished local training for Round '
              f'{self.current_round+1}/{self.rounds} in {epoch_time:.2f}s')
        
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

        elapsed = time.time() - self._last_timestamp
        self.cumulative_time += elapsed
        print(f'>>> Round {self.current_round+1} training took {elapsed:.2f}s; '
              f'cumulative so far: {self.cumulative_time:.2f}s')
        
        # reset for next round
        self._last_timestamp = time.time()


        self.average_loss = sum(input.loss for input in inputs) / len(inputs)
        self.aggregated_model_accuracy = sum(
            input.agg_validation_score for input in inputs) / len(inputs)
        self.local_model_accuracy = sum(
            input.local_validation_score for input in inputs) / len(inputs)
        print(f'Average aggregated model validation values = {self.aggregated_model_accuracy}')
        print(f'Average training loss = {self.average_loss}')
        print(f'Average local model validation values = {self.local_model_accuracy}')

        # pick off model name and metric
        # filenames (reuse logic from start)
        model_name  = self.model.__class__.__name__
        metric_acc  = "aggregated_model_accuracy"
        metric_time = "cumulative_training_time"
        metric_loss = "average_training_loss"

        if "Sketch" in model_name:
            try:
                q = self.model.conv1[0].q
            except Exception:
                q = self.model.conv1.q
            q_str = f"{q:g}".replace('.', 'p')
            fname_acc  = f"{model_name}_q{q_str}_{metric_acc}.txt"
            fname_time = f"{model_name}_q{q_str}_{metric_time}.txt"
            fname_loss = f"{model_name}_q{q_str}_{metric_loss}.txt"
        else:
            fname_acc  = f"{model_name}_{metric_acc}.txt"
            fname_time = f"{model_name}_{metric_time}.txt"
            fname_loss = f"{model_name}_{metric_loss}.txt"

        # append aggregated accuracy
        with open(fname_acc, "a") as f_acc:
            f_acc.write(
                f"Round {self.current_round+1}/{self.rounds}: "
                f"{self.aggregated_model_accuracy:.6f}\n"
            )

        # append cumulative time
        with open(fname_time, "a") as f_time:
            f_time.write(
                f"Round {self.current_round+1}/{self.rounds}: "
                f"{self.cumulative_time:.2f}s\n"
            )

        # append average training loss
        with open(fname_loss, "a") as f_loss:
            f_loss.write(
                f"Round {self.current_round+1}/{self.rounds}: "
                f"{self.average_loss:.6f}\n"
            )
        

        self.model = FedAvg([input.model for input in inputs])
        self.optimizer = [input.optimizer for input in inputs][0]
        self.current_round += 1
        if self.current_round < self.rounds:
            self.next(self.aggregated_model_validation,
                      foreach='collaborators', exclude=['private'])
        else:
            self.next(self.end)

    @aggregator
    def end(self):
        print(f'This is the end of the flow')

# Setup participants
aggregator = Aggregator()
aggregator.private_attributes = {}

# Setup collaborators with private attributes
collaborator_names = ['Portland', 'Seattle', 'Chandler','Bangalore']
collaborators = [Collaborator(name=name) for name in collaborator_names]

# Setup collaborators with private attributes
for idx, collaborator in enumerate(collaborators):
    # split whichever dataset we loaded above
    local_train = deepcopy(full_train)
    local_test  = deepcopy(full_test)

    local_train.data    = full_train.data[idx::len(collaborators)]
    local_train.targets = full_train.targets[idx::len(collaborators)]
    local_test.data     = full_test.data[idx::len(collaborators)]
    local_test.targets  = full_test.targets[idx::len(collaborators)]

    collaborator.private_attributes = {
        'train_loader': torch.utils.data.DataLoader(
            local_train, batch_size=batch_size_train, shuffle=False),
        'test_loader': torch.utils.data.DataLoader(
            local_test,  batch_size=batch_size_test,  shuffle=False),
    }
    collaborator.private_attributes = {
            'train_loader': torch.utils.data.DataLoader(local_train,batch_size=batch_size_train, shuffle=True),
            'test_loader': torch.utils.data.DataLoader(local_test,batch_size=batch_size_train, shuffle=True)
    }

local_runtime = LocalRuntime(aggregator=aggregator, collaborators=collaborators, backend='single_process')
print(f'Local runtime collaborators = {local_runtime.collaborators}')

model = None
best_model = None
optimizer = None
flflow = FederatedFlow(model, optimizer, rounds=100, checkpoint=False)
flflow.runtime = local_runtime
flflow.run()

print(f'\nFinal aggregated model accuracy for {flflow.rounds} rounds of training: {flflow.aggregated_model_accuracy}')