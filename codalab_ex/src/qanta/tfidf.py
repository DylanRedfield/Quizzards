import spacy
from spacy.tokenizer import Tokenizer

import pickle
from typing import List, Dict, Iterable, Optional, Tuple, NamedTuple
import os
from allennlp.modules.elmo import Elmo, batch_to_ids

options_file = "options_file.json"
weight_file = "weights_file.json"
import json
import torch

class Question(NamedTuple):
    qanta_id: int
    text: str
    first_sentence: str
    tokenizations: List[Tuple[int, int]]
    answer: str
    page: Optional[str]
    fold: str
    gameplay: bool
    category: Optional[str]
    subcategory: Optional[str]
    tournament: str
    difficulty: str
    year: int
    proto_id: Optional[int]
    qdb_id: Optional[int]
    dataset: str

    def to_json(self) -> str:
        return json.dumps(self._asdict())

    @classmethod
    def from_json(cls, json_text):
        return cls(**json.loads(json_text))

    @classmethod
    def from_dict(cls, dict_question):
        return cls(**dict_question)

    def to_dict(self) -> Dict:
        return self._asdict()

    @property
    def sentences(self) -> List[str]:
        """
        Returns a list of sentences in the question using preprocessed spacy 2.0.11
        """
        return [self.text[start:end] for start, end in self.tokenizations]

    def runs(self, char_skip: int) -> Tuple[List[str], List[int]]:
        """
        A Very Useful Function, especially for buzzer training.
        Returns runs of the question based on skipping char_skip characters at a time. Also returns the indices used
        q: name this first united states president.
        runs with char_skip=10:
        ['name this ',
         'name this first unit',
         'name this first united state p',
         'name this first united state president.']
        :param char_skip: Number of characters to skip each time
        """
        char_indices = list(range(char_skip, len(self.text) + char_skip, char_skip))
        return [self.text[:i] for i in char_indices], char_indices


class QantaDatabase:
    def __init__(self, split):
        '''
        split can be {'train', 'dev', 'test'} - gets both the buzzer and guesser folds from the corresponding data file.
        '''
        dataset_path = os.path.join('../../', 'qanta.' + split + '.json')
        with open(dataset_path) as f:
            self.dataset = json.load(f)

        self.version = self.dataset['version']
        self.raw_questions = self.dataset['questions']
        self.all_questions = [Question(**q) for q in self.raw_questions]
        self.mapped_questions = [q for q in self.all_questions if q.page is not None]

        self.guess_questions = [q for q in self.mapped_questions if q.fold == 'guess' + split]
        self.buzz_questions = [q for q in self.mapped_questions if q.fold == 'buzz' + split]


class QuizBowlDataset:
    def __init__(self, *, guesser=False, buzzer=False, split='train'):
        """
        Initialize a new quiz bowl data set
        guesser = True/False -> to use data from the guesser fold or not
        buzzer = True/False -> to use data from the buzzer fold or not
        split can be {'train', 'dev', 'test'}
        Together, these three parameters (two bools and one str) specify which specific fold's data to return - 'guesstrain'/'buzztrain'/'guessdev'/'buzzdev'/'guesstest'/'buzztest'
        """
        super().__init__()
        if not guesser and not buzzer:
            raise ValueError('Requesting a dataset which produces neither guesser or buzzer training data is invalid')

        if guesser and buzzer:
            print('Using QuizBowlDataset with guesser and buzzer training data, make sure you know what you are doing!')

        self.db = QantaDatabase(split)
        self.guesser = guesser
        self.buzzer = buzzer

    def data(self):
        questions = []
        if self.guesser:
            questions.extend(self.db.guess_questions)
        if self.buzzer:
            questions.extend(self.db.buzz_questions)

        return questions


class ElmoGuesser:

    def __init__(self):
        self.question_matrix = None
        self.answers = []
        self.i_to_ans = None
        self.elmo = Elmo(options_file, weight_file, num_output_representations=1)
        nlp = spacy.load('en')
        self.tokenizer = Tokenizer(nlp.vocab)

    def train(self, training_data):

        '''
        Must be passed the training data - list of questions from the QuizBowlDataset class
        '''
        print("train")

        # We want questions to store each question tokenized by word
        # and answers stored as a list
        questions = []
        for ques in training_data:
            tokens = self.tokenizer(' '.join(ques.sentences))
            tokens_list = [token.text for token in tokens]
            questions.append(tokens_list)
            self.answers.append(ques.page)

        print("chars to ids")
        character_ids = batch_to_ids(questions)
        print("elmo output")
        elmo_output = self.elmo(character_ids)

        # index at zero because we only have a single output representation
        word_embeddings = elmo_output['elmo_representations'][0]

        print("mean")
        # A matrix of size (num_train_questions * embed_length)
        self.question_matrix = word_embeddings.mean(1)

        print("train done")

    def guess(self, questions: List[str], max_n_guesses: Optional[int]) -> List[List[Tuple[str, float]]]:
        # Tokenize questions, get question embedding, compare them to given

        print("guess", len(questions))
        tokenized = []
        for ques in questions:
            tokens = self.tokenizer(ques)
            tokens_list = [token.text for token in tokens]
            tokenized.append(tokens_list)

        character_ids = batch_to_ids(tokenized)
        elmo_output = self.elmo(character_ids)
        word_embeddings = elmo_output['elmo_representations'][0]

        # A matrix size (num_input_questions * embed_length)
        question_embeddings = word_embeddings.mean(1)

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
                row.append([(self.answers[idx], max_values[i, j].item())])
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
            params = pickle.load(f)
            guesser = ElmoGuesser()
            guesser.question_matrix = params['question_matrix']
            guesser.answers = params['answers']

            guesser.elmo = Elmo(options_file, weight_file, num_output_representations=1)
            nlp = spacy.load('en')
            guesser.tokenizer = Tokenizer(nlp.vocab)

            return guesser

        print('Elmo Guesser -> load')


def train(device):
    """
    Train the tfidf model, requires downloaded data and saves to models/
    """
    dataset = QuizBowlDataset(guesser=True)
    elmo_guesser = ElmoGuesser()
    elmo_guesser.train(dataset.data())
    elmo_guesser.save()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train(device)
