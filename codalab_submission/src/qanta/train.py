# Standard library imports
import torch
# Third party imports

# Local application imports
# TODO: Could condense below, but to make sure it definitely works
try:
    import elmo
except:
    from . import elmo
try:
    import buzzer_rnn
except:
    from . import buzzer_rnn
try:
    import dataset as data_set
except:
    from . import dataset as data_set
try:
    from config import *
except:
    from .config import *

'''
    FILE: train.py 
    This allows for training and saving models without using the docker cli command
    Prints config specifications
'''

def guess_data():
    # TODO: OLIVER: CAN YOU INSERT THE CHOICE TO USE QANTA, SQUAD, and QANTA and SQAD TOGETHER?
    guess_dataset = data_set.QuizBowlDataset(guesser=True).data()

    return guess_dataset

def buzz_data():
    # TODO: OLIVER: CAN YOU INSERT THE CHOICE TO USE QANTA, SQUAD, and QANTA and SQAD TOGETHER?
    buzz_dataset = data_set.QuizBowlDataset(buzzer=True).data()

    return buzz_dataset

def elmo_train(device):
    """
    Train and save the elmo model
    - specified by the config model
    - called by either the main function of train.py or test.py train()

    data_set: specified data from either QANTA, SQUAD, TRIVIAQA or a mix
    """
    data = guess_data()
    elmo_guesser = elmo.ElmoGuesser(device)
    elmo_guesser.train(data, device)
    elmo_guesser.save(ELMO_MODEL_PATH)


def buzz_rnn_train(device):
    '''
    Grabs the data and trains the rnn buzzer and saves
    :param device:
    '''
    data = buzz_data(buzzer=True)
    buzzer = buzzer_rnn.RNNBuzzer()
    buzzer.train(data, device)
    buzzer.save(BUZZER_MODEL_PATH)

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if TRAIN_TYPE == 'elmo':
        print('Starting Training Elmo')
        elmo_train(device)
    else:
        print('Configure TRAIN_TYPE in config.py')
    if BUZZ_TYPE == 'rnn':
        print('Started Training RNNBuzzer')
        buzz_rnn_train(device)

    print('--TRAINING COMPLETE--')
    print('TRAIN_TYPE: ', TRAIN_TYPE)
    print('BUZZ_TYPE: ', BUZZ_TYPE)
    print('BUZZ_NUM_GUESSES', BUZZ_NUM_GUESSES)
    if BUZZ_TYPE == 'simple':
        print('BUZZ_THRESHOLD: ', BUZZ_THRESHOLD)
