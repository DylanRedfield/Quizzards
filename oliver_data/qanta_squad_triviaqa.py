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
import re

# --- Constants
QANTA_DIR = "../"
SQUAD_DIR = "../"
TRIVIAQA_DIR = "../qa/"

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

	def getTuple(self):
		return [self.text, self.answer]

class SquadQuestion:
	text: str
	answer: str

	def __init__(self, ctx: str, qa):
		if qa['is_impossible'] == True:
			self.answer = str(qa['plausible_answers'][0]['text'])
		else:
			self.answer = str(qa['answers'][0]['text'])

		question = qa['question']
		self.text = question

		for sentence in re.split('(?<=[.!?])', ctx):
			if self.answer not in sentence:
				self.text += str(sentence)

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

	def getTuple(self):
		return [self.text, self.answer]

class Dataset:
	def __init__(self, split='train', sources=['qanta']):
		'''
		split can be {'train', 'dev', 'test'} - gets both the buzzer and guesser folds from the corresponding data file.
		'''
		self.qanta_questions = []
		self.squad_questions = []
		self.triviaqa_questions = []
		self.questions = []

		self.printed = False

		if 'qanta' in sources:
			dataset_path = os.path.join(QANTA_DIR, 'qanta.' + split + '.json')
			with open(dataset_path) as f:
				self.qanta_dataset = json.load(f)

			self.qanta_version = self.qanta_dataset['version']
			self.qanta_raw_questions = self.qanta_dataset['questions']
			self.qanta_all_questions = [QantaQuestion(**q) for q in self.qanta_raw_questions]
			self.qanta_questions = [q.getTuple() for q in self.qanta_all_questions if q.page is not None]

		if 'squad' in sources:
			dataset_path = os.path.join(SQUAD_DIR, 'squad.' + split + '.json')
			with open(dataset_path) as f:
				self.squad_dataset = json.load(f)

			self.squad_version = self.squad_dataset['version']
			self.squad_data = self.squad_dataset['data']
			self.squad_paragraphs = [data['paragraphs'] for data in self.squad_data]

			for item in self.squad_paragraphs:
				for p in item:
					qas = p['qas']
					ctx = p['context']
					for qa in qas:
						# if not self.printed:
						# 	print('THIS THE ONE ------------------------------------------------------------------')
						# 	print(qa)
						# 	print('')
						# 	print(ctx)
						# 	self.printed = True
						self.squad_questions.append(SquadQuestion(ctx, qa).getTuple())

		# if 'triviaqa' in sources:
		# 	dataset_path = os.path.join(TRIVIAQA_DIR, 'squad.' + split + '.json') # this is wrong

		self.questions = self.qanta_questions + self.squad_questions + self.triviaqa_questions

		print('\t' + split, 'qanta:\t\t', len(self.qanta_questions))
		print('\t' + split, 'squad:\t\t', len(self.squad_questions))
		print('\t' + split, 'triviaqa:\t\t', len(self.triviaqa_questions))


	def data(self):
		return self.questions
