import argparse
import sys

import torch
from torch import nn
import click

from data import mnist
from model import MyAwesomeModel


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
@click.option("--optimizer", default=None, help='optimizer to use for training')
@click.option("--epochs", default=5, help='epochs to use for training')
# @click.option("--criterion", default=nn.NLLLoss, help='criterion to use for training')
# @click.option("--print_every", default=40, help='step for printing training results')

def train(lr, optimizer, epochs):#, criterion, print_every):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    trainloader, _ = mnist()

    criterion = nn.NLLLoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for e in range(epochs):
        model.train()
        for images, labels in trainloader:
            # Flatten images into a 784 long vector
            images.resize_(images.size()[0], 784)
            
            optimizer.zero_grad()
            
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

    checkpoint = {'input_size': 784,
              'output_size': 10,
              'hidden_layers': [each.out_features for each in model.hidden_layers],
              'state_dict': model.state_dict()}
    
    torch.save(checkpoint, 's1_development_environment/exercise_files/final_exercise/checkpoint.pth')


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    checkpoint = torch.load(model_checkpoint)
    model = MyAwesomeModel(checkpoint['input_size'],
                             checkpoint['output_size'],
                             checkpoint['hidden_layers'])
    model.load_state_dict(checkpoint['state_dict'])
    _, testloader = mnist()

    criterion = nn.NLLLoss()
    accuracy = 0
    test_loss = 0
    for images, labels in testloader:

        images = images.resize_(images.size()[0], 784)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ## Calculating the accuracy 
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        equality = (labels.data == ps.max(1)[1])
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    print("Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
    "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
