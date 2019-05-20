# Standard library imports
import os
from typing import List, Dict, Iterable, Optional, Tuple
# Third party imports
import spacy
from spacy.tokenizer import Tokenizer
from allennlp.modules.elmo import Elmo, batch_to_ids
import pickle

# Local application imports
try:
    from config import *
except:
    from .config import *


class ElmoGuesser:
    def __init__(self, device):
        self.question_matrix = None
        self.answers = []
        self.i_to_ans = None
        try:
            self.elmo = Elmo(OPTIONS_FILE, WEIGHTS_FILE, num_output_representations=1)
        except:
            self.elmo = Elmo(OPTIONS_FILE2, WEIGHTS_FILE2, num_output_representations=1)
        self.elmo = self.elmo.to(device)
        nlp = spacy.load('en_core_web_sm')
        self.tokenizer = Tokenizer(nlp.vocab)

    def train(self, training_data, device):

        '''
        Must be passed the training data - list of questions from the QuizBowlDataset class
        '''
        
        print("Training Elmo")
        print('DEVICE: ', device)

        # We want questions to store each question tokenized by word
        # and answers stored as a list
        questions = []
        for ques in training_data:
            tokens = self.tokenizer(' '.join(ques.sentences))
            tokens_list = [token.text for token in tokens]
            questions.append(tokens_list)
            self.answers.append(ques.page)

        print("chars to ids")
        character_ids = batch_to_ids(questions).to(device)
        # character_ids = character_ids.to(device)

        print("elmo output")
        elmo_output = self.elmo(character_ids)

        # index at zero because we only have a single output representation
        word_embeddings = elmo_output['elmo_representations'][0].to(device)

        print("mean")
        # A matrix of size (num_train_questions * embed_length)
        self.question_matrix = word_embeddings.mean(1).to(device)

        print("train done")

    def guess(self, questions: List[str], max_n_guesses: Optional[int]) -> List[List[Tuple[str, float]]]:
        # Tokenize questions, get question embedding, compare them to given

        print("guess", len(questions))
        tokenized = []
        for ques in questions:
            tokens = self.tokenizer(ques)
            tokens_list = [token.text for token in tokens]
            tokenized.append(tokens_list)

        character_ids = batch_to_ids(tokenized).to(device)

        elmo_output = self.elmo(character_ids)
        word_embeddings = elmo_output['elmo_representations'][0].to(device)

        # A matrix size (num_input_questions * embed_length)
        question_embeddings = word_embeddings.mean(1).to(device)

        # Matrix multiplication to find similarities between the rows of the training and input questions
        # into a matrix size (num_input_questions * num_train_questions)
        guess_matrix = self.question_matrix.mm(question_embeddings.t()).t()

        # Find the max values in each row which will corespond to the most similar training question
        # each is a vector size (num_input_questions)
        max_values, max_indicies = guess_matrix.topk(max_n_guesses, 1)

        # So now we habe a vector for each input question and we want to find the most similar saved question
        guesses = []

        for i in range(len(questions)):
            row = []
            for j in range(len(max_indicies[i])):
                idx = max_indicies[i][j]
                # guesses.append([(self.answers[j], guess_matrix[i, j]) for j in idxs])
                row.append((self.answers[idx], max_values[i, j].item()))
            guesses.append(row)

        return guesses

    def save(self, path):

        with open(path, 'wb') as f:
            pickle.dump({
                'question_matrix': self.question_matrix,
                'answers': self.answers
            }, f)

    def load(self, path):
        with open(path, 'rb') as f:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            params = pickle.load(f)
            guesser = ElmoGuesser(device)
            guesser.question_matrix = params['question_matrix']
            guesser.answers = params['answers']

            try:
                guesser.elmo = Elmo(OPTIONS_FILE, WEIGHTS_FILE, num_output_representations=1)
            except:
                guesser.elmo = Elmo(OPTIONS_FILE2, WEIGHTS_FILE2, num_output_representations=1)
            gueser.elmo = guesser.elmo.to(device)
            nlp = spacy.load('en_core_web_sm')
            guesser.tokenizer = Tokenizer(nlp.vocab)
            print('Elmo Guesser -> load')

            return guesser

        
