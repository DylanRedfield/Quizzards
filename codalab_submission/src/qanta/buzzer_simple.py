try:
    from config import *
except:
    from .config import *

def guess(guesses):
    '''
    Tells the buzzer to guess or not
    :param guesses: top n guesses from X model
    :return: True or False
    '''
    scores = [guess[1] for guess in guesses]
    buzz = scores[0] / sum(scores) >= BUZZ_THRESHOLD

    return buzz
