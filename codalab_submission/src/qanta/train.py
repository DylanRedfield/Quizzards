import torch
import elmo
import dataset as data_set
from config import *

'''
    FILE: train.py 
    This allows for training and saving models without using the docker cli command
'''

def data():
    # OLIVER: CAN YOU INSERT THE CHOICE TO USE QANTA, SQUAD, and QANTA and SQAD TOGETHER?
    dataset = data_set.QuizBowlDataset(guesser=True)

    return dataset

def elmo_train(device):
    """
    Train the tfidf model, requires downloaded data and saves to models/
    """
    dataset = data()
    elmo_guesser = elmo.ElmoGuesser()
    elmo_guesser.train(dataset.data())
    elmo_guesser.save()


if __name__ == '__main__':


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if TRAIN_TYPE == 'elmo':
        elmo_train(device)
    else:
        print('Configure TRAIN_TYPE in config.py')
