import torch
from torch import nn
import torch.optim as optim
import evaluator
import sklearn
import sklearn.metrics
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau 
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(net,trainloader,validationloader,n_epochs=10,lr=0.001):
    net.to(device)
    writer = SummaryWriter()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    iteration = 0
    for epoch in range(n_epochs):
        running_loss = 0.0
        print_every = 500
        total_loss = 0.0
        examples = 0
        for i, sample in enumerate(trainloader, 0):
            net.train()
            inputs = sample['image']
            labels = sample['identity']
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()

            outputs,embedding = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if iteration % 1000 == 0:
                validate(net,validationloader,iteration,writer)
            
            # print statistics
            running_loss += loss.item()
            total_loss += loss.item()
            writer.add_scalar("MeanLoss/iteration",loss.item(),iteration)
            iteration +=1
            examples += 1
            if (i % print_every) == (print_every-1):
                print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/print_every))
                running_loss = 0.0
        if isinstance(net,nn.DataParallel):
            net.module.save_weights("model.mdl")
        else:
            net.save_weights("model.mdl")
        writer.add_scalar("MeanLoss/epoch",total_loss/examples,epoch)
        print("Total Loss= ",total_loss/examples)

    print('Finished Training')
    writer.close()



def validate(net,loader,iteration,writer):
    net.eval()
    with torch.no_grad():
        running_loss = 0
        iterations = 0
        for i, sample in enumerate(loader, 0):
            inputs = sample['image']
            labels = sample['identity']
            inputs, labels = inputs.to(device), labels.to(device)

            outputs,embedding = net(inputs)
            loss = (labels.eq(torch.argmax(outputs,dim=1).long())).sum()
            running_loss += loss.item()
            iterations += outputs.shape[0]
        writer.add_scalar("ValidationAccuracy/iteration",running_loss/iterations,iteration)
        return running_loss/iterations

def train_triplet(net, trainloader,generator_loader, n_epochs=10, lr=0.001):
    net.to(device)
    writer = SummaryWriter()

    target, distances = generate_predictions(net, generator_loader,True)

    threshold, auc = find_threshold(target, distances)
    writer.add_scalar("Auc/epoch", auc, 0)
    print("threshold=", threshold)
    print("Auc=", auc)

    with open('threshold.txt', 'w+') as f:
         f.write(str(threshold))

    accuracy = evaluator.evaluate_model()
    print("Accuracy=", accuracy)
    writer.add_scalar("Accuracy/epoch", accuracy, 0)


    optimizer = optim.Adam(net.parameters(), lr=lr)
    iteration = 0
    margin = 1.0
    criterion = nn.TripletMarginLoss(margin=20.0)
    for epoch in range(n_epochs):
        running_loss = 0.0
        print_every = 1000
        total_loss = 0.0
        examples = 0
        
        for i, sample in enumerate(trainloader, 0):
            net.train()
            anchor = sample['anchor']
            positive = sample['positive']
            negative = sample['negative']
            anchor, positive,negative = anchor.to(device), positive.to(device),negative.to(device)

            optimizer.zero_grad()

            _, embedding_anchor = net(anchor, True)
            _, embedding_positive = net(positive, True)
            _, embedding_negative = net(negative, True)
            loss = criterion(embedding_anchor, embedding_positive,embedding_negative)

            loss.backward()
            optimizer.step()
            # print statistics

            running_loss += loss.item()
            total_loss += loss.item()
            
            iteration += 1
            examples += 1
            if (i % print_every) == (print_every - 1):
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / print_every))
                writer.add_scalar("MeanTripletLoss/iteration", running_loss/print_every, iteration//print_every)
                running_loss = 0.0

        if isinstance(net, nn.DataParallel):
            net.module.save_weights("model.mdl")
        else:
            net.save_weights("model.mdl")
        writer.add_scalar("MeanTripletLoss/epoch", total_loss / examples, epoch+1)

        print("Total Loss= ", total_loss / examples)

        target, distances = generate_predictions(net, generator_loader,True)

        threshold, auc = find_threshold(target, distances)
        writer.add_scalar("Auc/epoch", auc, epoch+1)
        print("threshold=", threshold)
        print("Auc=", auc)

        with open('threshold.txt', 'w+') as f:
            f.write(str(threshold))

        accuracy = evaluator.evaluate_model()
        print("Accuracy=", accuracy)
        writer.add_scalar("Accuracy/epoch", accuracy, epoch+1)


    print('Finished Training')
    writer.close()


def generate_predictions(net,loader,triplets = False):
    net.eval()
    with torch.no_grad():
        all_distances = []
        all_target = []
        for i, sample in enumerate(loader, 0):
            if i > 60000:
                break
            net.train()
            inputs = sample['image']
            labels = sample['identity']
            inputs, labels = inputs.to(device), labels.to(device)

            _, embedding = net(inputs,triplets)

            original = embedding[0]
            original_label = labels[0]
            other_labels = labels[1:]
            others = embedding[1:]

            targets = (other_labels == original_label).cpu().numpy()

            for target in targets:
                all_target.append(target)
            distances = torch.cdist(original.view(1, -1), others.view(others.shape[0], -1))
            for distance in distances.cpu().numpy():
                for d in distance:
                    all_distances.append(d)

        return all_target,all_distances




def find_threshold(target, predicted):
    target = np.asarray(target)
    predicted = np.asarray(predicted)
    
    fpr, tpr, threshold = sklearn.metrics.roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.ix[(roc.tf-0).abs().argsort()[:1]]
    
    return list(roc_t['threshold'])[0],sklearn.metrics.roc_auc_score(target,predicted)

