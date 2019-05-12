import argparse
import pandas as pd
import numpy as np
import time
import os
from typing import List, Dict, Iterable, Optional, Tuple, NamedTuple
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_


from typing import List, Optional, Tuple
from collections import defaultdict
import pickle
import json
from os import path
from . import util

import click
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, jsonify, request

TFIDF_MODEL_PATH = 'tfidf.pickle'
BUZZER_MODEL_PATH = 'buzzer.pickle'
BUZZ_NUM_GUESSES = 10
BUZZ_THRESHOLD = 0.3

GUESSER_TRAIN_FOLD = 'guesstrain'
BUZZER_TRAIN_FOLD = 'buzztrain'
TRAIN_FOLDS = {GUESSER_TRAIN_FOLD, BUZZER_TRAIN_FOLD}

# Guesser and buzzers produce reports on these for cross validation
GUESSER_DEV_FOLD = 'guessdev'
BUZZER_DEV_FOLD = 'buzzdev'
DEV_FOLDS = {GUESSER_DEV_FOLD, BUZZER_DEV_FOLD}

# System-wide cross validation and testing
GUESSER_TEST_FOLD = 'guesstest'
BUZZER_TEST_FOLD = 'buzztest'

DS_VERSION = '2018.04.18'

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
    def __init__(self, dataset_path=os.path.join('data', util.QANTA_MAPPED_DATASET_PATH)):
    # def __init__(self, dataset_path=os.path.join('../../data', f'qanta.mapped.{DS_VERSION}.json')):
        with open(dataset_path) as f:
            self.dataset = json.load(f)

        self.version = self.dataset['version']
        self.raw_questions = self.dataset['questions']
        self.all_questions = [Question(**q) for q in self.raw_questions]
        self.mapped_questions = [q for q in self.all_questions if q.page is not None]

        self.train_questions = [q for q in self.mapped_questions if q.fold in TRAIN_FOLDS]
        self.guess_train_questions = [q for q in self.train_questions if q.fold == GUESSER_TRAIN_FOLD]
        self.buzz_train_questions = [q for q in self.train_questions if q.fold == BUZZER_TRAIN_FOLD]

        self.dev_questions = [q for q in self.mapped_questions if q.fold in DEV_FOLDS]
        self.guess_dev_questions = [q for q in self.dev_questions if q.fold == GUESSER_DEV_FOLD]
        self.buzz_dev_questions = [q for q in self.dev_questions if q.fold == BUZZER_DEV_FOLD]

        self.buzz_test_questions = [q for q in self.mapped_questions if q.fold == BUZZER_TEST_FOLD]
        self.guess_test_questions = [q for q in self.mapped_questions if q.fold == GUESSER_TEST_FOLD]

class QuizBowlDataset:
    def __init__(self, *, guesser=False, buzzer=False ):
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

        self.db = QantaDatabase()
        self.guesser = guesser
        self.buzzer = buzzer

    def data(self):
        '''
        Returns the questions - where each question is an object of the Question class - according to the specific fold specified by the split, guesser, buzzer parameters.
        '''
        questions = []
        if self.guesser:
            questions.extend(self.db.guess_train_questions)
        if self.buzzer:
            questions.extend(self.db.buzz_train_questions)

        return questions

class QuestionDataset(Dataset):
    """
    Pytorch data class for questions
    """

    ###You don't need to change this funtion
    def __init__(self, examples):
        self.questions = []
        self.labels = []

        for qq, ll in examples:
            self.questions.append(qq)
            self.labels.append(ll)

    ###You don't need to change this funtion
    def __getitem__(self, index):
        return self.questions[index], self.labels[index]

    ###You don't need to change this funtion
    def __len__(self):
        return len(self.questions)

class RNNBuzzer(nn.Module):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    We use a LSTM for our buzzer.
    """

    #### You don't need to change the parameters for the model

    # n_input represents dimensionality of each specific feature vector, n_output is number of labels
    def __init__(self, n_input=10, n_hidden=50, n_output=2, dropout=0.5):
        super(RNNBuzzer, self).__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.n_output = n_output

        ### Your Code Here ---
        # define lstm layer, going from input to hidden. Remember to have batch_first=True.
        self.lstm = nn.LSTM(n_input, n_hidden, batch_first=True)

        # define linear layer going from hidden to output.
        self.hidden_to_label = nn.Linear(n_hidden, n_output)

        # self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # ---you can add other things like dropout between two layers, but do so in forward function below,
        # as we have to perform an extra step on the output of LSTM before going to linear layer(s).
        # The MODEL FOR TEST CASES is just single layer LSTM followed by 1 linear layer - do not add anything else for the purpose of passing test cases!!

    def forward(self, X):
        # get the batch size and sequence length (max length of the batch)
        # dim of X: batch_size x batch_max_len x input feature vec dim
        # batch_size, seq_len, _ = X.size()
        print('X.size()', X.size())

        ###Your code here --
        # Get the output of LSTM - (output dim: batch_size x batch_max_len x lstm_hidden_dim)

        lstm_out, lstm_hidden = self.lstm(X)

        # reshape (before passing to linear layer) so that each row contains one token
        # essentially, flatten the output of LSTM
        # dim will become batch_size*batch_max_len x lstm_hidden_dim

        lstm_out = lstm_out.contiguous().view(-1, self.n_hidden)

        # Get logits from the final linear layer
        logits = self.hidden_to_label(lstm_out)

        return logits

def loss_fn(outputs, labels):
    # to compute cross entropy loss
    outputs = F.log_softmax(outputs, dim=1)

    # reshape labels to give a flat vector of length batch_size*seq_len
    labels = labels.contiguous()
    labels = labels.view(-1)

    # flatten all predictions
    outputs = outputs.contiguous()
    outputs = outputs.view(-1, 2)  # 2 is the number of labels

    # mask out 'PAD' tokens
    mask = (labels > -1).float()

    # the number of tokens is the sum of elements in mask
    num_tokens = int(torch.sum(mask).data)

    # pick the values corresponding to labels and multiply by mask
    outputs = outputs[range(outputs.shape[0]), labels] * mask

    # cross entropy loss for all non 'PAD' tokens
    return -torch.sum(outputs) / num_tokens

def batchify(batch):
    """
    Gather a batch of individual examples into one batch,
    which includes the question feature vec, question length, and labels
    (feature vec and labels generated in create_feature_vecs_and_labels function)
    """

    question_len = list()
    label_list = list()
    for ex in batch:
        question_len.append(len(ex[0]))
        label_list.append(ex[1])

    '''
    Padding the labels - unequal length sequences for sequenial data like we have. 
    Since actual labels are 0/1 - we pad with -1, and will use this when 'masking' labels during loss and
    accuracy evaluation.
    '''
    target_labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(y) for y in label_list], padding_value=-1).t()

    # dimension is dimension of every feature vector = n_guesses in this homework setting
    dim = batch[0][0].shape[1]

    # similar padding happens for the feature vectors, with vector of all zeros appended.
    x1 = torch.FloatTensor(len(question_len), max(question_len), dim).zero_()
    for i in range(len(question_len)):
        question_feature_vec = batch[i][0]
        vec = torch.FloatTensor(question_feature_vec)
        x1[i, :len(question_feature_vec)].copy_(vec)
    q_batch = {'feature_vec': x1, 'len': torch.FloatTensor(question_len), 'labels': target_labels}
    return q_batch

def create_feature_vecs_and_labels(guesses_and_scores, answers, n_guesses):
    xs, ys = [], []

    # print('guesses_and_scores: ', guesses_and_scores)
    for i in range(len(answers)):
        guesses_scores = guesses_and_scores[i]
        ans = answers[i]
        length = len(ans)
        labels = []
        prob_vec = []

        for j in range(length):
            ## YOUR CODE BELOW
            temp_prob = []
            for k in range(n_guesses):
                temp_prob.append(guesses_scores[j][k][1])
            if ans[j] == guesses_scores[j][0][0]:
                labels.append(1)
            else:
                labels.append(0)
            prob_vec.append(temp_prob)

        xs.append(np.array(prob_vec))

    # print('xs: ', xs)
    # print('ys: ', ys)

    exs = list(zip(xs, ys))
    return exs

def create_feature_vecs_and_labels_altered(guesses_and_scores, answers, n_guesses):
    xs, ys = [], []

    print('guesses_and_scores: ', guesses_and_scores)
    for i in range(len(answers)):
        guesses_scores = guesses_and_scores[i]
        ans = answers[i]
        length = len(ans)
        labels = []
        prob_vec = []

        for j in range(length):
            ## YOUR CODE BELOW
            temp_prob = []
            for k in range(n_guesses):
                temp_prob.append(guesses_scores[j][k][1])
            if ans[j] == guesses_scores[j][0][0]:
                labels.append(1)
            else:
                labels.append(0)
            prob_vec.append(temp_prob)

        xs.append(np.array(prob_vec))

    # print('xs: ', xs)
    # print('ys: ', ys)

    # print('xs: ', xs)
    exs = list(xs)
    return exs

def create_quess(text):
    question = {}
    question['qanta_id'] = 0
    question['text'] = text
    question['first_sentence'] = ""
    question['tokenizations'] =  [[0, 0]] # List[Tuple[int, int]]
    question['answer'] = ""
    question['page'] = "" # Optional[str]
    question['fold'] = "" # : str
    question['gameplay'] = True # ???: bool
    question['category'] = "" #: Optional[str]
    question['subcategory'] = "" # : Optional[str]
    question['tournament'] = "" # : str
    question['difficulty'] = "" #: str
    question['year'] = 0 # : int
    question['proto_id'] = 0 # Optional[int]
    question['qdb_id']  = 0 #: Optional[int]
    question['dataset'] = "" # : str

    return question

def generate_ques_data_for_guesses(questions, char_skip=50):
    ques_nums = []
    char_indices = []
    question_texts = []
    answers = []
    question_lens = []
    # print("Preparing Data for Guessing; # of questions: " + str(len(questions)))
    # questions = Question().to_dict(questions)
    for q in questions:
        # print('Q type: ', type(q))
        # print('Q class: ', q.__class__)
        if q.__class__ != Question:
            q = create_quess(q)
            q = Question(**q)
        qnums_temp, answers_temp, char_inds_temp, curr_ques_texts = [], [], [], []
        for text_run, char_ix in zip(*(q.runs(char_skip))):
            curr_ques_texts.append(text_run)
            qnums_temp.append(q.qanta_id)
            answers_temp.append(q.page)
            char_inds_temp.append(char_ix)
        ques_nums.append(qnums_temp)
        char_indices.append(char_inds_temp)
        question_texts.append(curr_ques_texts)
        answers.append(answers_temp)
        question_lens.append(len(curr_ques_texts))

    return ques_nums, answers, char_indices, question_texts, question_lens

def generate_guesses_and_scores(model, questions, max_guesses, char_skip=50):
    # print('TYPE IN GEN GUESS SCORES-questions : ', type(questions))
    # print('TYPE IN GEN GUESS SCORES-questions[0] : ', type(questions[0]))
    # get the neccesary data
    qnums, answers, char_indices, ques_texts, ques_lens = generate_ques_data_for_guesses(questions, char_skip)
    # print('Guessing...')
    # print('TYPE IN GEN GUESS SCORES-ques_texts : ', type(ques_texts))
    # print('TYPE IN GEN GUESS SCORES-ques_texts[0] : ', type(ques_texts[0]))
    # print('ques_texts: ', ques_texts[0])
    guesses_and_scores = []
    for i in range(0, len(ques_texts), 250):
        try:
            q_texts_temp = ques_texts[i:i + 250]
            q_lens = ques_lens[i:i + 250]
        except:
            q_texts_temp = ques_texts[i:]
            q_lens = ques_lens[i:]

        # flatten
        q_texts_flattened = []
        for q in q_texts_temp:
            q_texts_flattened.extend(q)

        # store guesses for the flattened questions
        # print('q_texts_flattened')
        print('q_texts_flattened: ', q_texts_flattened)
        flattened_guesses_scores = model.guess(q_texts_flattened, max_guesses)
        print('flattened_guesses_scores: ', flattened_guesses_scores)

        # de-flatten using question lengths, and add guesses and scores
        # (now corresponding to one question at a time) to the main list
        j = 0
        for k in q_lens:
            guesses_and_scores.append(flattened_guesses_scores[j:j + k])
            j = j + k

    assert len(guesses_and_scores) == len(ques_texts)
    # print('guesses_and_scores[0] ', guesses_and_scores[0])

    # print('Done Generating Guesses and Scores.')

    return qnums, answers, char_indices, ques_texts, ques_lens, guesses_and_scores

def guess_and_buzz(tfidf_guesser, buzzer_guesser, question_text) -> Tuple[str, bool]:
    print('GUESS AND BUZZ RAN')
    print('questiion_text: ', question_text)
    guesses = tfidf_guesser.guess([question_text], BUZZ_NUM_GUESSES)[0]

    print('GUESSES: ', guesses)

    buzz_guess = buzzer_guesser.guess(tfidf_guesser, [question_text])
    # scores = [guess[1] for guess in guesses]
    # buzz = scores[0] / sum(scores) >= BUZZ_THRESHOLD

    return guesses[0][0], buzz_guess

def batch_guess_and_buzz(tfidf_guesser, buzzer_guesser, questions) -> List[Tuple[str, bool]]:
    question_guesses = tfidf_guesser.guess(questions, BUZZ_NUM_GUESSES)
    # question_buzzes = buzzer_guesser.guess(tfidf_guesser, [questions])
    outputs = []
    for guesses, question in zip(question_guesses, questions):
        # scores = [guess[1] for guess in guesses]
        # buzz = buzzer_guesser.guess([question_text], tfidf_guesser)
        buzz = buzzer_guesser.guess(tfidf_guesser, [question])
        outputs.append((guesses[0][0], buzz))
    return outputs

class BuzzerGuesser:
    def __init__(self):
        self.rnn_model = RNNBuzzer()

    def train(self, training_data, tfidf_guesser) -> None:
        # train_buzz_questions = QuizBowlDataset(guesser_train=True).data()
        train_buzz_questions = training_data
        n_guesses = 10
        char_skip = 50
        batch_size = 8
        num_epochs = 25
        train_qnums, train_answers, train_char_indices, train_ques_texts, train_ques_lens, train_guesses_and_scores = \
            generate_guesses_and_scores(tfidf_guesser, train_buzz_questions, n_guesses, char_skip=char_skip)

        train_exs = create_feature_vecs_and_labels(train_guesses_and_scores, train_answers, n_guesses)
        # print(len(train_exs))

        train_dataset = QuestionDataset(train_exs)

        train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0,
                                  collate_fn=batchify)

        for epoch in range(num_epochs):
            self.rnn_model.train()
            optimizer = torch.optim.Adam(self.rnn_model.parameters())
            print_loss_total = 0
            epoch_loss_total = 0

            #### modify the following code to complete the training funtion

            # best_train_acc, best_dev_acc = 0.0, 0.0

            for idx, batch in enumerate(train_loader):
                question_feature_vec = batch['feature_vec']
                question_len = batch['len']
                labels = batch['labels']

                #### Your code here ----

                # zero out
                self.rnn_model.zero_grad()

                # get output from model
                logits = self.rnn_model.forward(question_feature_vec, question_len)

                # use loss_fn defined above to calculate loss

                loss = loss_fn(logits, labels)

                optimizer.step()

                # backprop
                loss.backward()

                ###Your code ends ---
                clip_grad_norm_(self.rnn_model.parameters(), 5)

    def guess(self, tfidf_guesser, questions: List[str]) -> List[List[Tuple[str, float]]]:
        char_skip = 50
        n_guesses = 10
        batch_size = 8

        print('guess-questions', type(questions))
        test_qnums, test_answers, test_char_indices, test_ques_texts, test_ques_lens, test_guesses_and_scores = \
            generate_guesses_and_scores(tfidf_guesser, questions, n_guesses, char_skip)

        # print('TEST ANSWERS: ', test_answers)

        test_exs = create_feature_vecs_and_labels_altered(test_guesses_and_scores, test_answers, n_guesses)
        # print('TEST_EXS 1 : ', test_exs)

        # test_dataset = QuestionDataset(test_exs)

        # test_dataset = QuestionDataset(test_exs)
        # test_sampler = torch.utils.data.sampler.SequentialSampler(test_dataset)
        # test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=0,
        #                          collate_fn=batchify)

        # test_exs = torch.FloatTensor(test_dataset.questions)
        # test_exs = torch.FloatTensor(test_exs)
        self.rnn_model.eval()
        # print('TYPE: test_exs: ', type(test_exs))
        # print('TEST_EXS: ', test_exs)

        temp_ex = np.asarray(test_exs)
        # print('TYPE: temp_exs: ', type(temp_ex))
        # print('SHAPE: ', temp_ex.shape)
        # test_exs = torch.FloatTensor(test_exs)
        test_exs = torch.from_numpy(temp_ex)
        # print('TYPE: test_exs: ', type(test_exs))
        # print('SIZE: test_exs: ', type(test_exs.size()))
        # print('TYPE: test_exs: ', type(test_exs.float()))
        test_exs = test_exs.float()


        # for idx, batch in enumerate(test_loader):
        #     question_feature_vec = batch['feature_vec']
        #     question_len = batch['len']


        logits = self.rnn_model(test_exs)
        print('LOGITS: ', logits)
        # logits = torch.sum(logits, dim=0)

        (first, second) = logits[0]



        if first > second:
            print('FIRST: ', first)
            return False
        else:
            print('SECOND: ', second)
            return True

        # logits = self.rnn_model(test_exs, len(test_exs))



    def save(self):
        with open(BUZZER_MODEL_PATH, 'wb') as f:
            pickle.dump({
                'rnn_model': self.rnn_model
            }, f)

    @classmethod
    def load(cls, tfidf_guesser):
        with open(BUZZER_MODEL_PATH, 'rb') as f:
            params = pickle.load(f)
            guesser = BuzzerGuesser()
            guesser.rnn_model = params['rnn_model']
            return guesser

class TfidfGuesser:
    '''
    This is the guesser class; here we use Tfidf.

    Methods:

    train: To train the Tfidf guesser.
    guess: Use the trained model to make guesses on a text of question.
    save: Save the trained model to location specified in GUESSER_MODEL_PATH.
    load: Can load a previously saved trained model from the location specified in GUESSER_MODEL_PATH.
    '''

    def __init__(self):
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.i_to_ans = None

    def train(self, training_data) -> None:
        '''
        Must be passed the training data - list of questions from the QuizBowlDataset class
        '''
        questions, answers = [], []
        for ques in training_data:
            questions.append(ques.sentences)
            answers.append(ques.page)

        answer_docs = defaultdict(str)
        for q, ans in zip(questions, answers):
            text = ' '.join(q)
            answer_docs[ans] += ' ' + text

        x_array = []
        y_array = []
        for ans, doc in answer_docs.items():
            x_array.append(doc)
            y_array.append(ans)

        self.i_to_ans = {i: ans for i, ans in enumerate(y_array)}

        # below hypermaters for the vectorizer may be tuned.
        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_df=.9).fit(x_array)

        self.tfidf_matrix = self.tfidf_vectorizer.transform(x_array)

    def guess(self, questions: List[str], max_n_guesses: Optional[int]) -> List[List[Tuple[str, float]]]:
        '''
        Will output the top n guesses specified by max_n_guesses acting on a list of question texts (strings).
        '''
        # print('TFIDF.GUESS-questions: ', type(questions))
        # print('TFIDF.GUESS-questions[0]: ', type(questions[0]))
        # print('TFIDF.GUESS-questions[1]: ', type(questions[1]))
        # training_examples = []
        training_pages = []
        # for q in questions:
        #     training_examples.append(q.sentences)
        #     training_pages.append(q.page)

        representations = self.tfidf_vectorizer.transform(questions)
        guess_matrix = self.tfidf_matrix.dot(representations.T).T
        guess_indices = (-guess_matrix).toarray().argsort(axis=1)[:, 0:max_n_guesses]

        guesses = []
        for i in range(len(questions)):
            idxs = guess_indices[i]
            guesses.append([(self.i_to_ans[j], guess_matrix[i, j]) for j in idxs])
        print('GUESS: ', guesses[0][0])
        return guesses

    def save(self):
        with open(TFIDF_MODEL_PATH, 'wb') as f:
            pickle.dump({
                'i_to_ans': self.i_to_ans,
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'tfidf_matrix': self.tfidf_matrix
            }, f)

    @classmethod
    def load(cls):
        with open(TFIDF_MODEL_PATH, 'rb') as f:
            params = pickle.load(f)
            guesser = TfidfGuesser()
            guesser.tfidf_vectorizer = params['tfidf_vectorizer']
            guesser.tfidf_matrix = params['tfidf_matrix']
            guesser.i_to_ans = params['i_to_ans']
            return guesser

def create_app(enable_batch=True):
    tfidf_guesser = TfidfGuesser.load()
    buzzer_guesser = BuzzerGuesser.load(tfidf_guesser)
    app = Flask(__name__)

    @app.route('/api/1.0/quizbowl/act', methods=['POST'])
    def act():
        question = request.json['text']
        guess, buzz = guess_and_buzz(tfidf_guesser, buzzer_guesser, question)
        return jsonify({'guess': guess, 'buzz': True if buzz else False})

    @app.route('/api/1.0/quizbowl/status', methods=['GET'])
    def status():
        return jsonify({
            'batch': enable_batch,
            'batch_size': 200,
            'ready': True,
            'include_wiki_paragraphs': False
        })

    @app.route('/api/1.0/quizbowl/batch_act', methods=['POST'])
    def batch_act():
        questions = [q['text'] for q in request.json['questions']]
        return jsonify([
            {'guess': guess, 'buzz': True if buzz else False}
            for guess, buzz in batch_guess_and_buzz(tfidf_guesser, buzzer_guesser, questions)
        ])
    return app

@click.group()
def cli():
    pass

@cli.command()
@click.option('--host', default='0.0.0.0')
@click.option('--port', default=4861)
@click.option('--disable-batch', default=False, is_flag=True)
def web(host, port, disable_batch):
    """
    Start web server wrapping tfidf model
    """
    app = create_app(enable_batch=not disable_batch)
    app.run(host=host, port=port, debug=False)

@cli.command()
def train():
    """
    Train the tfidf model, requires downloaded data and saves to models/
    """
    # dataset = QuizBowlDataset(guesser=True)
    # tfidf_guesser = TfidfGuesser()
    #
    # tfidf_guesser.train(dataset.data())
    # tfidf_guesser.save()

    try:
        tfidf_guesser = TfidfGuesser.load()
        print('TFIDF GUESSER LOADED')
    except:
        dataset = QuizBowlDataset(guesser=True)
        tfidf_guesser = TfidfGuesser()

        tfidf_guesser.train(dataset.data())
        tfidf_guesser.save()

    try:
        buzzer_guesser = BuzzerGuesser.load(tfidf_guesser)
        print('BUZZER GUESSER LOADED')
    except:
        buzzer_dataset = QuizBowlDataset(buzzer=True)
        buzzer_guesser = BuzzerGuesser()
        buzzer_guesser.train(buzzer_dataset.data(), tfidf_guesser)
        buzzer_guesser.save()

    # question = request.json['text']
    # guess, buzz = guess_and_buzz(tfidf_guesser, buzzer_guesser, question)
    # return jsonify({'guess': guess, 'buzz': True if buzz else False})

    question = "A lapse function and a shift vector field are used in a numerical approach to solving these equations called the ADM formalism. The linearized form of these equations is in terms of the d'Alembertian (dal-ahm-BARE-shin) of a tensor often represented with an \"h\" that satisfies the harmonic coordinate condition. In geometrized units, this equation is \"big-G-sub-mu-nu plus lambda little-g-sub-mu-nu equals 8-pi-T-sub-mu-nu.\" This set of ten nonlinear, partial differential equations is often presented as a tensor equation. This equation's namesake called one of its terms, the cosmological constant, the biggest blunder of his life, but now astrophysicists interpret that term of these equations as dark energy. For 10 points, name these eponymous equations used to describe the interaction of energy and spacetime curvature in general relativity."
    guess_and_buzz(tfidf_guesser, buzzer_guesser, question)

    # buzzer_dataset = QuizBowlDataset(buzzer=True)
    # buzzer_guesser = BuzzerGuesser()
    # buzzer_guesser.train(buzzer_dataset.data(), tfidf_guesser)
    # buzzer_guesser.save()

@cli.command()
@click.option('--local-qanta-prefix', default='data/')
@click.option('--retrieve-paragraphs', default=False, is_flag=True)
def download(local_qanta_prefix, retrieve_paragraphs):
    """
    Run once to download qanta data to data/. Runs inside the docker container, but results save to host machine
    """
    util.download(local_qanta_prefix, retrieve_paragraphs)


if __name__ == '__main__':
    cli()
