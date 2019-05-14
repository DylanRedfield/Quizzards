import torch
import elmo
import util
import dataset as data_set

def data():
    dataset = data_set.QuizBowlDataset(guesser=True)

    return dataset

def train(device):
    """
    Train the tfidf model, requires downloaded data and saves to models/
    """
    dataset = data()
    elmo_guesser = elmo.ElmoGuesser()
    elmo_guesser.train(dataset.data())
    elmo_guesser.save()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train(device)
