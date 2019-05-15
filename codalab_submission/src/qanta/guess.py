# Standard library imports
from typing import List, Dict, Iterable, Optional, Tuple
# Third party imports\
import torch
# Local application imports
try:
    import elmo
except:
    from . import elmo
try:
    import buzzer_simple
except:
    from . import buzzer_simple
try:
    import buzzer_rnn
except:
    from . import buzzer_rnn
try:
    from config import *
except:
    from .config import *



def guess(question) -> Tuple[str, bool]:
    '''
    Guesses and buzzes based on guesser model and buzz type found in config file
    :param question: text that is of type Question in the dataset.py
    :return: top guess and to buzz or not to buzz
    '''

    if TRAIN_TYPE == 'elmo':
        guess_model = elmo.ElmoGuesser()
        guess_model.load(ELMO_MODEL_PATH)
    else:
        print('Configure TRAIN_TYPE in config.py')



    guesses = guess_model.guess([question], BUZZ_NUM_GUESSES)[0]

    if BUZZ_TYPE == 'simple':
        buzz = buzzer_simple.guess(guesses)
    elif BUZZ_TYPE == 'rnn':
        buzzer = buzzer_rnn.RNNBuzzer()
        buzzer.load()
        buzz = buzzer.guess(guess_model, question)
    else:
        print('Configure BUZZ_TYPE in config.py')

    return guesses[0][0], buzz

def batch_guess(questions) -> List[Tuple[str, bool]]:
    '''
        Guesses and buzzes based on guesser model and buzz type found in config file in a batch
        :param question: list of text that is of type Question in the dataset.py
        :return: top guess and to buzz or not to buzz
        '''
    outputs = []

    if TRAIN_TYPE == 'elmo':
        guess_model = elmo.ElmoGuesser()
    else:
        print('Configure TRAIN_TYPE in config.py')

    guess_model.load()

    question_guesses = guess_model.guess(questions, BUZZ_NUM_GUESSES)

    for guesses in question_guesses:
        # TODO: THIS MIGHT BE WRONG ...
        if BUZZ_TYPE == 'simple':
            buzz = buzzer_simple.guess(guesses)
        elif BUZZ_TYPE == 'rnn':
            # TODO: NOT SURE IF GUESSES OR QUESTION SHOULD GO IN HERE...
            buzz = buzzer_rnn.RNNBuzzer.guess(guess_model, question)
        else:
            print('Configure BUZZ_TYPE in config.py')
        outputs.append((guesses[0][0], buzz))

    return outputs


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    question = "Relative"
    print('--Starting Guessing--')

    guess, buzz = guess(question)

    print('--GUESSING COMPLETE--')
    print('TRAIN_TYPE: ', TRAIN_TYPE)
    print('BUZZ_TYPE: ', BUZZ_TYPE)
    print('BUZZ_NUM_GUESSES', BUZZ_NUM_GUESSES)
    if BUZZ_TYPE == 'simple':
        print('BUZZ_THRESHOLD: ', BUZZ_THRESHOLD)
