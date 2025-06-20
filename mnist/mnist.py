# adapted from https://github.com/pytorch/examples/blob/main/mnist/main.py
# run from base directory with `python -m mnist.mnist`
import os


import argparse
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import wandb
from src.SillyLayers import SillyLinear, SillyConv2d, stepgen

from src.RandumbTensor import CreateRandumbTensor
from src.utils import getBack

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128, bias=False)
        self.fc2 = nn.Linear(128, 10, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class CastedLinear(SillyLinear):

    def __init__(self, in_features, out_features, int_dim_gen, seed_gen):
        super().__init__(in_features, out_features, int_dim_gen, seed_gen, bias=False, rt_bias=True)

    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype))
    
class SillyNet(nn.Module):
    def __init__(self, int_dim_gen, seed_gen):
        super(SillyNet, self).__init__()
        self.conv1 = SillyConv2d(1, 32, int_dim=int_dim_gen, seed=seed_gen, kernel_size=3, stride=1, rt_bias=True)
        #self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, device="cuda")
        self.conv2 = SillyConv2d(32, 64, int_dim=int_dim_gen, seed=seed_gen, kernel_size=3, stride=1, rt_bias=True)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, device="cuda")
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # self.fc1 = nn.Linear(9216, 128)
        self.fc1 = CastedLinear(9216, 128, int_dim_gen=int_dim_gen, seed_gen=seed_gen)
        
        # self.fc2 = nn.Linear(128, 10)
        self.fc2 = CastedLinear(128, 10, int_dim_gen=int_dim_gen, seed_gen=seed_gen)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch, train_losses):
    model.train()
    # train_accumulation_steps = 4
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        
        # if batch_idx % train_accumulation_steps == 0:
        #     for p in model.parameters():
        #         p.grad /= train_accumulation_steps
                
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
            train_losses.append(loss.detach().item())
            if args.wandb:
                # log the scale_bias terms
                log_dict = {}
                for name, buf in model.named_parameters():
                    if name.endswith("_scale_bias"):
                        log_dict[name] = buf.detach().item()
                log_dict["train_loss"] = loss.detach().item()
                wandb.log(log_dict)
            if args.dry_run:
                break


def test(args, model, device, test_loader, test_accs):
    model.eval()
    test_loss = 0 
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accs.append(100. * correct / len(test_loader.dataset))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    if args.wandb: wandb.log({"test_loss": test_loss, "test_acc": 100. * correct / len(test_loader.dataset)})


def main():
    # import lovely_tensors as lt
    # lt.monkey_patch()
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    # note: dry-run default set to True for debugging, should be False
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--wandb', action='store_true', default=True,
                    help='Include wandb logging')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)
    training_args = {"is_rand_model": True, "int_dim_offs": 128, "int_dim_step": 0, "seed_offs": 5, "seed_step": 4}
    
    if args.wandb:
        run = wandb.init(
            project="mnist",  # Specify your project
            config=dict(vars(args), **training_args),  # convert args dataclass into dict and pass in all the configs
        )

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)


    if training_args["is_rand_model"]:
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            # transforms.ConvertImageDtype(torch.bfloat16)
            ])
    else:
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    if training_args["is_rand_model"]:
        # generators for int_dim and seed
        int_dim_gen = stepgen(training_args["int_dim_offs"], training_args["int_dim_step"])
        seed_gen = stepgen(training_args["seed_offs"], training_args["seed_step"])

        model = SillyNet(int_dim_gen, seed_gen)
        model = model.cuda().float()
        for m in model.modules():
            if isinstance(m, CastedLinear):
                # m.to(torch.float32)
                m.float()
    else:
        model = Net().to(device)
        model = torch.compile(model)
        
    if args.wandb: wandb.watch(model, log="all")
    
    # for module in model.modules():
    #     print("converting to real")
    #     if hasattr(module, 'weight') and hasattr(module.weight, 'get_materialized') and callable(module.weight.get_materialized):
    #         module.weight = nn.Parameter(module.weight.get_materialized())
    #         if module.bias is not None: module.bias = nn.Parameter(module.bias.get_materialized())
            
    print("parameters optimized:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.device, param.dtype)
            # if param.dim() > 1:
            #     print("2d xavier")
            #     nn.init.xavier_normal_(param, gain=)
            # else:
            #     print("1d xavier")
            #     nn.init.normal_(param)
            # lt.plot(param, center="mean").fig.savefig(f"{os.path.abspath(os.path.dirname(__file__))}/weight_dists/rand_model_leakyrelu/{name}.png")

    for buf in model.buffers():
        print(type(buf), buf.size(), buf.device)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    train_losses = []
    test_accs = []
    test(args, model, device, test_loader, test_accs)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, train_losses)
        test(args, model, device, test_loader, test_accs)
        scheduler.step()

    # NOTE: initial loss should be ln(10) = 2.30 
    fig, axs = plt.subplots(2)
    fig.suptitle(f'mnist train results, final acc {test_accs[-1]}%')
    axs[0].plot(train_losses)
    axs[0].set_title("train loss")
    axs[1].plot(test_accs)
    axs[1].set_title("eval test accs")
    fig.savefig('bench_out/mnist.png')

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
        
    if args.wandb: run.finish()


if __name__ == '__main__':
    main()