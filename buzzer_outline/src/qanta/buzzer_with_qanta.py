import argparse
import nltk
import json
import pandas as pd
import numpy as np
import time
import os
from os import path
from typing import List, Dict, Iterable, Optional, Tuple, NamedTuple
from collections import defaultdict
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import random


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_

# --- My imports
# BERT
#from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
# logging for BERT
import logging
logging.basicConfig(level=logging.INFO)
# regex
import re

# --- Data creation
class QantaQuestion(NamedTuple):
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

	def __str__(self):
		return 'text: ' + self.text + '\nanswer: ' + self.answer

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

		def runs_word(self, word_skip: int) -> List[str]:
			#TODO fix this
			toks = tokenizer.tokenize(self.text)

class SquadQuestion:
	text: str
	answer: str

	def __init__(self, ctx: str, qa):
		if qa['is_impossible'] == True:
			self.answer = qa['plausible_answers'][0]['text']
		else:
			self.answer = qa['answers'][0]['text']

		question = qa['question']
		self.text = question
		for sentence in re.split('(?<=[.!?])',ctx):
			if self.answer not in sentence:
				self.text += sentence

	def __str__(self):
		return 'text: ' + self.text + '\nanswer: ' + self.answer

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
		return [sentence for sentence in re.split('(?<=[.!?])', self.text)]

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

	def runs_word(self, word_skip: int) -> List[str]:
		#TODO fix this
		toks = tokenizer.tokenize(self.text)

class Dataset:
	def __init__(self, split='train', sources=['qanta']):
		'''
		split can be {'train', 'dev', 'test'} - gets both the buzzer and guesser folds from the corresponding data file.
		'''
		self.qanta_questions = []
		self.squad_questions = []
		self.questions = []

		if 'qanta' in sources:
			dataset_path = os.path.join('..', 'qanta.' + split + '.json')
			with open(dataset_path) as f:
				self.qanta_dataset = json.load(f)

			self.qanta_version = self.qanta_dataset['version']
			self.qanta_raw_questions = self.qanta_dataset['questions']
			self.qanta_all_questions = [QantaQuestion(**q) for q in self.qanta_raw_questions]
			self.qanta_questions = [q for q in self.qanta_all_questions if q.page is not None]

		if 'squad' in sources:
			dataset_path = os.path.join('..', 'squad.' + split + '.json')
			with open(dataset_path) as f:
				self.squad_dataset = json.load(f)

			# TODO: figure this out
			self.squad_version = self.squad_dataset['version']
			self.squad_data = self.squad_dataset['data']
			self.squad_paragraphs = [data['paragraphs'] for data in self.squad_data]

			#print(self.squad_paragraphs[0])

			self.squad_questions = []
			for item in self.squad_paragraphs:
				for p in item:
					qas = p['qas']
					ctx = p['context']
					for qa in qas:
						self.squad_questions.append(SquadQuestion(ctx, qa))

		self.questions = self.qanta_questions + self.squad_questions

		print('\t' + split, 'qanta:\t\t', len(self.qanta_questions))
		print('\t' + split, 'squad:\t\t', len(self.squad_questions))

	def data(self):
		return self.questions

# --- Tokenizer
# Load pre-trained model tokenizer (vocabulary)
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# --- Guesser


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Options for RNN Buzzer')

	parser.add_argument('--n_guesses', type=int, default=10,
						help='number of top guesses to consider for creating feature vector')

	# guesser options
	parser.add_argument('--guesser_train_limit', type=int, default=-1,
						help='Limit the data used to train the Guesser')
	parser.add_argument('--guesser_dev_limit', type=int, default=1500,
						help='Limit the dev data used to evaluate the Guesser')
	parser.add_argument('--guesser_test_limit', type=int, default=2000,
						help='Limit the test data used to evaluate the Guesser')
	parser.add_argument('--guesser_batch_size', type=int, default=8, help='Batch Size for the training of the guesser')
	parser.add_argument('--guesser_num_epochs', type=int, default=25, help='Number of Epochs for training the guesser')
	parser.add_argument('--guesser_saved_flag', type=bool, default=False,
						help='flag indicating use of saved guesser model or training one')
	parser.add_argument('--guesser_see_test_accuracy', type=bool, default=False,
						help='flag indicating seeing test result FINALLY after tuning on dev')
	parser.add_argument('--guesser_checkpoint', type=int, default=50)
	parser.add_argument('--guesser_model_path', type=str, default='bert_guesser.pickle',
						help='path for saving the trained guesser model')

	# buzzer options
	parser.add_argument('--buzzer_train_limit', type=int, default=5000,
						help='Limit the data used to train the Buzzer')
	parser.add_argument('--buzzer_dev_limit', type=int, default=500,
						help='Limit the dev data used to evaluate the buzzer (total is 1161)')
	parser.add_argument('--buzzer_test_limit', type=int, default=500,
						help='Limit the test data used to evaluate the buzzer (total is 1953)')
	parser.add_argument('--buzzer_batch_size', type=int, default=8, help='Batch Size for the training of the buzzer')
	parser.add_argument('--buzzer_num_epochs', type=int, default=25, help='Number of Epochs for training the buzzer')
	parser.add_argument('--buzzer_saved_flag', type=bool, default=False,
						help='flag indicating using saved data files to train buzzer, or generating them')
	parser.add_argument('--buzzer_see_test_accuracy', type=bool, default=False,
						help='flag indicating seeing test result FINALLY after tuning on dev')
	parser.add_argument("--char_skip", type=int, default=50,
						help='number of characters to skip after which buzzer should predict')
	parser.add_argument('--buzzer_checkpoint', type=int, default=50)
	parser.add_argument('--buzzer_model_path', type=str, default='bert_guesser.pickle',
						help='path for saving the trained guesser model')

	args = parser.parse_args()

	# create datasets

	print('\nINFO ABOUT DATASETS')

	train_data = Dataset(split='train', sources=['qanta', 'squad']).data()
	print('\ttotal train data:\t',len(train_data), '\n')
	dev_data = Dataset(split='dev', sources=['qanta']).data()
	print('\ttotal dev data:\t\t',len(dev_data), '\n')
	test_data = Dataset(split='test', sources=['qanta']).data()
	print('\ttotal test data:\t',len(test_data))

	if args.guesser_train_limit < 0:
		train_guess_questions = train_data[:-args.buzzer_train_limit]
	else:
		train_guess_questions = train_data[:args.guesser_train_limit]

	if args.guesser_dev_limit < 0:
		dev_guess_questions = dev_data[:-args.buzzer_dev_limit]
	else:
		dev_guess_questions = dev_data[:args.guesser_dev_limit]

	if args.guesser_test_limit < 0:
		test_guess_questions = test_data[:-args.buzzer_test_limit]
	else:
		test_guess_questions = test_data[:args.guesser_test_limit]

	train_buzz_questions = train_data[-args.buzzer_train_limit:]

	dev_buzz_questions = dev_data[-args.buzzer_dev_limit:]

	test_buzz_questions = test_data[-args.buzzer_test_limit:]

	print('\n\ttrain_guess_questions:\t', len(train_guess_questions),
		  '\n\tdev_guess_questions:\t', len(dev_guess_questions),
		  '\n\ttest_guess_questions:\t', len(test_guess_questions))
	print('\n\ttrain_buzz_questions:\t', len(train_buzz_questions),
		  '\n\tdev_buzz_questions:\t', len(dev_buzz_questions),
		  '\n\ttest_buzz_questions:\t', len(test_buzz_questions))

	# train guesser

	'''if args.guesser_saved_flag:
		guesser_model = BertGuesser().load(args.guesser_model_path)
	else:
		guesser_model = get_trained_guesser_model(train_guess_questions)
		guesser_model.save(args.guesser_model_path)
		print('Guesser Model Saved! Use --guesser_saved_flag=True when you next run the code to load the trained guesser directly.')
	'''

	'''if args.buzz_data_saved_flag:
		train_exs = np.load('train_exs.npy')
		dev_exs = np.load('dev_exs.npy')
		test_exs = np.load('test_exs.npy')
		dev_ques_texts = np.load('dev_ques_texts.npy')
	else:
		print('Generating Guesses for Training Buzzer Data')
		train_qnums, train_answers, train_char_indices, train_ques_texts, train_ques_lens, train_guesses_and_scores = \
			generate_guesses_and_scores(guesser_model, train_buzz_questions, args.n_guesses, char_skip=args.char_skip)
		print('Generating Guesses for Dev Buzzer Data')
		dev_qnums, dev_answers, dev_char_indices, dev_ques_texts, dev_ques_lens, dev_guesses_and_scores = \
			generate_guesses_and_scores(guesser_model, dev_buzz_questions, args.n_guesses, char_skip=args.char_skip)
		# print('DEV QUES[0]', dev_ques_texts[0])
		print('Generating Guesses for Test Buzzer Data')
		test_qnums, test_answers, test_char_indices, test_ques_texts, test_ques_lens, test_guesses_and_scores = \
			generate_guesses_and_scores(guesser_model, test_buzz_questions, args.n_guesses, char_skip=args.char_skip)
		train_exs = create_feature_vecs_and_labels(train_guesses_and_scores, train_answers, args.n_guesses)
		# print(len(train_exs))
		dev_exs = create_feature_vecs_and_labels(dev_guesses_and_scores, dev_answers, args.n_guesses)
		# print(len(dev_exs))
		test_exs = create_feature_vecs_and_labels(test_guesses_and_scores, test_answers, args.n_guesses)
		# print(len(test_exs))
		np.save('train_exs.npy', train_exs)
		np.save('dev_exs.npy', dev_exs)
		np.save('test_exs.npy', test_exs)
		np.save('dev_ques_texts.npy', dev_ques_texts)
		print('The Examples for Train, Dev, Test have been SAVED! Use --buzz_data_saved_flag=True next time when you \
				run the code to use saved data and not generate guesses again.')

	train_dataset = QuestionDataset(train_exs)
	train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=0,
							  collate_fn=batchify)

	dev_dataset = QuestionDataset(dev_exs)
	dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
	dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, sampler=dev_sampler, num_workers=0,
							collate_fn=batchify)

	test_dataset = QuestionDataset(test_exs)
	test_sampler = torch.utils.data.sampler.SequentialSampler(test_dataset)
	test_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler, num_workers=0,
							 collate_fn=batchify)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	model = RNNBuzzer()
	model.to(device)
	print(model)

	for epoch in range(args.num_epochs):
		print('start epoch %d' % (epoch + 1))
		train_acc, dev_acc = train(args, model, train_loader, dev_loader, device)

	if args.see_test_accuracy:
		print('The Final Test Set Accuracy')
		print(evaluate(test_loader, model, device))'''
