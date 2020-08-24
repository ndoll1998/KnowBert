
import re
import json
import time
import numpy as np
from functools import reduce
from collections import OrderedDict
from stanfordnlp import Pipeline
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

def get_token_spans(sentence, words):

	spans = []
	begin = 0
	for w in words:
		# remove all whitespaces
		while not sentence.startswith(w.text):
			# if text[0] != " ":
			assert sentence[0] == " "
			begin += 1
			sentence = sentence[1:]
		# add span
		spans.append((begin, begin + len(w.text)))
		# update begin and text for next token
		begin += len(w.text)
		sentence = sentence[len(w.text):]

	return spans

class ConceptParser:

	def __init__(self):
		# load stopwords
		self.stop_words = stopwords.words('english')
		# conjunctions to consider
		self.conjugation_tokens = ['and', 'or']
		# create lemmatizer and nlp-pipeline
		self.lemmatizer = WordNetLemmatizer()
		self.tokenizer = Pipeline(lang='en', processors="tokenize")
		self.nlp = Pipeline(lang='en', processors="tokenize,pos,depparse", tokenize_pretokenized=True)
		# map dep-type to function
		self.func_map = {
			'nsubj':	nsubject,
			'det':		det,
			'dep':		dep,
			'dobj':		dobj,
			'acomp':	acomp,
			'amod':		amod,
			'aux':		aux,
			'nn':		nn,
			'neg':		neg,
			'prep':		prep,
		}

	def remove_PRP_without_NN(self, words, deps):
		# check all dependencies
		for i, d in zip(range(len(deps)-1, -1, -1), deps[::-1]):
			# get part-of-speech tags of dependency-targets
			pos_tags = [d[0].pos, d[2].pos]
			# check condition
			if ("PRP" in pos_tags) and ("NN" not in pos_tags) and (d[1] != 'nsubj'):
				del deps[i]
		# return dependencies
		return deps

	def process_sentence(self, sentence):
		# get words and dependencies
		words = sentence.words
		deps = sentence.dependencies
		# remove unnecessary dependencies
		deps = self.remove_PRP_without_NN(words, deps)

		# all words except stopwords are concepts
		# this differs from usual idea of concepts
		word_concepts = [(w,) for w in words if w.text not in self.stop_words]
		# get concepts from each depencency
		dep_concepts = (self.func_map[t](w1, w2) for (w1, t, w2) in deps if t in self.func_map)
		dep_concepts = [concept for concept in dep_concepts if concept is not None]
		# get conjugations concepts
		conj_positions = self.conjugation_finder(words)
		conj_concepts = sum(map(lambda i: self.conjugator(words, i), conj_positions), [])
		# get manual concepts
		munual_concepts = self.manual(words)

		# throw all together and return
		concepts = set(word_concepts + dep_concepts + conj_concepts + munual_concepts)
		return list(concepts)

	def parse(self, sentence):
		# tokenize sentence
		doc = self.tokenizer(sentence)
		all_words = sum([sent.words for sent in doc.sentences], [])
		# get word spans
		all_word_spans = get_token_spans(sentence, all_words)
		# apply lemmatizer on all words and
		# reconstruct document from tokens such that nlp pipeline 
		# recreates the exact sentences and tokens
		tokenized_sentence = '\n'.join([' '.join([self.lemmatizer.lemmatize(w.text) for w in sent.words]) for sent in doc.sentences])
		# apply pipeline
		doc = self.nlp(tokenized_sentence)
		# apply word-spans to words
		all_words = sum([sent.words for sent in doc.sentences], [])
		assert len(all_words) == len(all_word_spans)
		for w, (b, e) in zip(all_words, all_word_spans):
			w.begin = b
			w.end = e
		# process single sentence
		return self.process_sentence(doc.sentences[0])

	# This rule has been created for "TO" type postags for relation between objects
	def manual(self, words):
		manual_concepts = []

		for i in range(1, len(words) - 1):
			word_span = (words[i-1], words[i], words[i+1])
			pos_span = words[i-1].pos + words[i].pos + words[i+1].pos
			
			if pos_span in ["JJTOVB", "JJTOVBD", "JJTOVBZ", "JJSTOVB", "JJSTOVBD", "JJSTOVBZ", "JJRTOVB", "JJRTOVBD", "JJRTOVBZ"]:
				manual_concepts.append(word_span)
		
		return manual_concepts

	# This rule has been created for finding the multiple positions of conjugations
	def conjugation_finder(self, words):
		# find all conjugations
		occ = sum(([i for i, w in enumerate(words) if w.text == t and w.pos == 'CC'] for t in self.conjugation_tokens), [])
		occ = sorted(occ)
		return occ

	# This rule has been created for "AND" types for relation between structures of sentence
	def conjugator(self, words, i):
		concepts = []

		word1 = i - 1
		word2 = min((j for j, w in enumerate(words[i+1:], start=i+1) if w.pos != 'DT'), default=-1)

		target_words = [word1] + ([word2] if word2 >= 0 else [])

		if len(target_words) == 2:
			concepts.append((words[word1], words[i], words[word2]))
		# find verb and noun
		verbs = list(filter(lambda i: words[i].pos == 'VB', range(i - 3, i)))
		nouns = list(filter(lambda i: words[i].pos == 'NN', range(i - 3, i)))
		# conjugation with noun
		if len(nouns) > 0:
			concepts.extend(
				[(words[j], words[nouns[0]]) for j in target_words if j != nouns[0]]
			)
		# conjugation with verb
		if len(verbs) > 0:
			concepts.extend(
				[(words[verbs[0]], words[j]) for j in target_words]
			)
		# relations after conjugation
		relations = ["between", "over", "with", "on", "to", "of", "into", "in", "at"]
		for j, w in enumerate(words[i:], start=i):
			if w.text in relations:
				word3 = j + 1
				concepts.extend(
					[(words[j], words[word3]) for j in target_words if j != word3]
				)
				break

		return concepts



""" Dependency Types """

# nsubj : nominal subject : Nominal subject is a noun phrase which is the syntactic subject of a clause
def nsubject(w1, w2):
	pos = [w1.pos, w2.pos]

	# DT check
	if "DT" not in pos:

		# NN and JJ check
		if "JJ" in pos:
			return (w1, w2)

		if "NN" in pos:								
			if "PRP" in pos:
				return (w1,)
			else:
				return (w2, w1)

	if "DT" in pos:
		return (w1,)

	return None				
	
# det : determiner : Determiner is the relation between the head of an NP and its determiner
def det(w1, w2):
	pos = [w1.pos, w2.pos]

	if "DT" not in pos:
		return (w2, w1)
	if "DT" in pos:
		return (w1,)

	return None

# dep : dependent : Dependency is labeled as dep when the system is unable to determine a more precise dependency relation between two words
def dep(w1, w2):
	pos = [w1.pos, w2.pos]

	if ("DT" not in pos) and ("JJ" in pos):
		return (w2, w1)

	if ("DT" not in pos) and ("JJ" not in pos):
		if ("NN" in pos) and ("VB" not in pos):
			return (w1,)
		else:
			return (w1, w2)

	if "DT" in pos:
		return (w1,)

	return None

# dobj : direct object : Direct object of a VP is the noun phrase which is the (accusative) object of the verb
def dobj(w1, w2):
	return (w1, w2)

# acomp : adjectival complement : Adjectival complement of a verb is an adjectival phrase which functions as the complement
def acomp(w1, w2):
	return (w1, w2)
	
# advmod : adverbial modifier : Adverbial modifier of a word is a (non-clausal) adverb or adverbial phrase (ADVP) that serves to modify the meaning of the word
def advmod(w1, w2):
	pos = [w1.pos, w2.pos]
	#print pos
	if ("VB" in pos) and ("JJ" in pos):
		return (w1, w2)
	if ("VB" in pos) and ("JJ" not in pos) and ("IN" in pos):
		return (w1, w2)
	if ("VB" in pos) and ("JJ" not in pos) and ("IN" not in pos):
		return (w2, w1)
	if "VB" not in pos:
		return (w2, w1)

	return None

# amod : adjectival modifier : Adjectival modifier of an NP is any adjectival phrase that serves to modify the meaning of the NP
def amod(w1, w2):
	pos = [w1.pos, w2.pos]

	if "VB" in pos:
		return (w1, w2)
	else:
		return (w2, w1)

# aux : auxiliary : Auxiliary of a clause is a non-main verb of the clause	
def aux(w1, w2):
	pos = [w1.pos, w2.pos]

	if "TO" in pos:
		return (w1,)
	if "VB" not in pos:
		return (w2, w1)

# nn : noun compound modifier : Noun compound modifier of an NP is any noun that serves to modify the head noun
def nn(w1, w2):
	# order words by index
	if w2.index < w1.index:
		return (w2, w1)
	else:
		return (w1, w2)

def neg(w1, w2):
	if w1 != w2:
		return (w2, w1)

# prep : prepositional modifier : Prepositional modifier of a verb, adjective, or noun is any prepositional phrase that serves to modify the meaning of the verb, adjective, noun, or even another prepositon
def prep(w1, w2):
	return (w1, w2)


if __name__ == '__main__':

	concept = ConceptParser()

	concepts = concept.parse("The coffee was hot and tasty.")

	for e, cs in concepts.items():
		print("%s: %s" % (e, str(cs)))
	
	
	
	# print(concept.parse("Redevelopment of the Darlington #nuclearplant shows some of the work we're famous for in numerous #nuclear projects."))

	# concepts_per_sentence = concept.parse_all([
		# "The coffee was hot and tasty.",
		# "I enjoyed the time i spent at this new restaurant!"
	# ])

	# for concepts in concepts_per_sentence:
		# print(concepts)