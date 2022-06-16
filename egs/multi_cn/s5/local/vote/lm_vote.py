#!/usr/bin/python3

import argparse
import jieba
import logging
import numpy
import os
import re
import shutil
import torch

from transformers import BertTokenizer, BertForMaskedLM


class Voter:
	def __init__(self, id, weight_function, backward):
		self.id = id
		self.weight_function = weight_function
		self.backward = backward
		self.model = None
		self.tokenizer = None

		self.init = False

	def init_by_list(self, transcription):
		# init by VoterFactory, always got forward text
		self.plain_text = ''.join(transcription[1:])

		self.weight = 1.0

		self.used_votes = []
		self.remain_votes = []

		if self.backward != False:
			self.plain_text = self.plain_text[::-1]

			# reverse all words and characters inside the words
			transcription[1:] = transcription[:0:-1]
			for i in range(1, len(transcription)):
				transcription[i] = transcription[i][::-1]

		self.remain_votes = list(transcription[1:])

		self.init = True

		logging.debug('init_by_list(%d), votes %s' % (self.id, self.remain_votes))
		return

	def init_by_string(self, sentence):
		# init by other Voter, may be backward text
		self.plain_text = sentence

		self.weight = 1.0

		self.used_votes = []
		self.remain_votes = []

		if self.backward != False:
			self.remain_votes = list(jieba.cut(self.plain_text[::-1], cut_all = False))

			# reverse all words and characters inside the words
			self.remain_votes.reverse()
			for i in range(0, len(self.remain_votes)):
				self.remain_votes[i] = self.remain_votes[i][::-1]
		else:
			self.remain_votes = list(jieba.cut(self.plain_text, cut_all = False))

		self.init = True

		logging.debug('init_by_string(%d), votes %s' % (self.id, self.remain_votes))
		return

	def set_model(self, model, tokenizer):
		if self.init == False:
			logging.error('set_model(%d): initialization not complete' % (self.id))
			return

		self.model = model
		self.tokenizer = tokenizer

		return

	def fork_voter(self, sentence):
		if self.init == False:
			logging.error('fork_voter(%d): initialization not complete' % (self.id))
			return None

		child = Voter(self.id, self.weight_function, self.backward)

		child.init_by_string(sentence)
		child.set_model(self.model, self.tokenizer)

		# use weight of complete transcription
		child.weight = self.weight

		logging.debug('fork_voter(%d), sentence %s' % (self.id, sentence))
		return child

	def update_weight(self):
		if self.init == False:
			logging.error('fork_voter(%d): initialization not complete' % (self.id))
			return False

		if self.weight_function == 'count':
			self.weight = 1.0
			return True

		if self.weight_function == 'ppl' or self.weight_function == 'ppl-count':
			ppl = self.calculate_ppl()
			self.weight = 1.0 / ppl

			if self.weight_function == 'ppl-count':
				self.weight += 1.0

			return True

		logging.error('update_weight(%d): unknown weight_function \'%s\'' % (self.id, self.weight_function))
		return False

	def calculate_ppl(self):
		ppl = float('inf')

		if self.init == False:
			logging.error('calculate_ppl(%d): initialization not complete' % (self.id))
			return ppl

		if self.model == None:
			logging.error('calculate_ppl(%d): no model available' % (self.id))
			return ppl

		if self.tokenizer == None:
			logging.error('calculate_ppl(%d): no tokenizer available' % (self.id))
			return ppl

		text = self.plain_text
		model = self.model
		tokenizer = self.tokenizer

		if self.backward != False:
			text = text[::-1]

		with torch.no_grad():
			tokenize_input = tokenizer.tokenize(text)
			tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
			sen_len = len(tokenize_input)

			if sen_len != 0:
				sentence_loss = 0.
				for i, word in enumerate(tokenize_input):
					# add mask to i-th character of the sentence
					tokenize_input[i] = '[MASK]'
					mask_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])

					output = model(mask_input)

					prediction_scores = output[0]
					softmax = torch.nn.Softmax(dim = 0)
					ps = softmax(prediction_scores[0, i]).log()
					word_loss = ps[tensor_input[0, i]]
					sentence_loss += word_loss.item()

					tokenize_input[i] = word

				ppl = numpy.exp(-sentence_loss/sen_len)

			logging.info('calculate_ppl(%d): ppl %f for text \'%s\'' % (self.id, ppl, text))
		return ppl

	def do_vote(self):
		if self.init == False:
			logging.error('do_vote(%d): initialization not complete' % (self.id))
			return

		if len(self.remain_votes) != 0:
			self.current_vote = self.remain_votes.pop(0)
		else:
			self.current_vote = '\n'

		future_votes = []
		# result doesn't look good so remove
		#for i in range(min(1, len(self.remain_votes))):
		#	future_votes.append(self.remain_votes[i])

		logging.info('do_vote(%d): current %s, used %s, future %s' % (self.id, repr(self.current_vote), self.used_votes, future_votes))
		return self.used_votes, self.current_vote, future_votes

	def purge_used_votes(self):
		if self.init == False:
			logging.error('purge_used_votes(%d): initialization not complete' % (self.id))
			return

		if len(self.used_votes) != 0:
			logging.info('do_align(%d): clean used_votes %s' % (self.id, self.used_votes))
			self.used_votes = []

		return

	def do_align(self, winner):
		if self.init == False:
			logging.error('do_align(%d): initialization not complete' % (self.id))
			return ''

		# case 1: already aligned with others, just clean up the used_votes queue
		if winner == self.current_vote:
			logging.info('do_align(%d): my vote wins' % (self.id))

			# align success, empty the queue
			self.purge_used_votes()
			return winner

		# case 2: my vote is a substring of winner, success to align
		#  ex. my vote: AB
		#      winner:  ABCD
		if len(winner) > len(self.current_vote):
			if self.current_vote == winner[:len(self.current_vote)]:
				new_vote = self.current_vote
				for i in range(len(self.remain_votes)):
					new_vote += self.remain_votes[i]

					min_len = min(len(winner), len(new_vote))

					if winner[:min_len] != new_vote[:min_len]:
						# not substring
						break

					if min_len != len(winner):
						# add one more vote to check
						continue

					logging.info('do_align(%d): my vote wins but too short' % (self.id))

					for j in range(i + 1):
						logging.info('do_align(%d): pop %s from remain_votes %s' % (self.id, repr(self.remain_votes[0]), self.remain_votes))
						self.remain_votes.pop(0)

					if min_len != len(new_vote):
						# insert remain of new_vote to remain_votes
						remain = new_vote[min_len:]

						logging.info('do_align(%d): push %s to remain_votes %s' % (self.id, repr(remain), self.remain_votes))
						self.remain_votes.insert(0, remain)

					# align success, empty the queue
					self.purge_used_votes()
					return winner

		# case 3: winner is a substring of my vote, success to align
		#  ex. my vote: ABCD
		#      winner:  AB
		if len(winner) < len(self.current_vote):
			if winner == self.current_vote[:len(winner)]:
				logging.info('do_align(%d): my vote wins but too long' % (self.id))

				# insert remain of current_vote to remain_votes
				remain = self.current_vote[len(winner):]

				logging.info('do_align(%d): push %s to remain_votes %s' % (self.id, repr(remain), self.remain_votes))
				self.remain_votes.insert(0, remain)

				# align success, empty the queue
				self.purge_used_votes()
				return winner

		# case 4: winner is in my used_votes, success to align
		new_vote = ''
		for i in range(len(self.used_votes)):
			new_vote += self.used_votes[i]

			idx = new_vote.find(winner)
			if idx < 0:
				continue

			logging.info('do_align(%d): winner in used_votes' % (self.id))

			for j in range(i + 1):
				logging.info('do_align(%d): pop %s from used_votes %s' % (self.id, repr(self.used_votes[0]), self.used_votes))
				self.used_votes.pop(0)

			if idx + len(winner) < len(new_vote):
				remain = new_vote[idx + len(winner):]

				logging.info('do_align(%d): push %s to used_votes %s' % (self.id, repr(remain), self.used_votes))
				self.used_votes.insert(0, remain)

			logging.info('do_align(%d): append current_vote %s to used_votes %s' % (self.id, repr(self.current_vote), self.used_votes))
			self.used_votes.append(self.current_vote)

			# insert back to remain_votes for next do_vote() call
			logging.info('do_align(%d): push used_votes %s to remain_votes %s' % (self.id, self.used_votes, self.remain_votes))
			self.used_votes.extend(self.remain_votes)
			self.remain_votes = self.used_votes

			# align success, empty the queue
			self.purge_used_votes()
			return winner

		# case 5: winner is in my remain_votes, success to align
		for i in range(min(len(self.used_votes) + 1, len(self.remain_votes))):
			if self.remain_votes[i] != winner:
				continue

			logging.info('do_align(%d): winner in remain_votes' % (self.id))

			# align myself
			while True:
				logging.info('do_align(%d): pop %s from remain_votes %s' % (self.id, repr(self.remain_votes[0]), self.remain_votes))
				if self.remain_votes.pop(0) == winner:
					break

			# align success, empty the queue
			self.purge_used_votes()
			return winner

		# case 6: fail to align, update used_votes
		if self.current_vote != '\n':
			logging.info('do_align(%d): fail to align, append current_vote %s to used_votes %s' % (self.id, repr(self.current_vote), self.used_votes))
			self.used_votes.append(self.current_vote)

		# seems not helping...
		#while len(self.used_votes) > 4:
		#	self.used_votes.pop(0)

		return self.current_vote

class VoterFactory:
	black_list = ['[SPK]', '[FIL]']

	def __init__(self, vote_dir, file_name, id, weight_function, backward):
		self.init = False

		data_dir = vote_dir + '/data'

		path = data_dir + '/' + file_name

		if os.path.exists(path) == False:
			logging.error('VoterFactory(%d): transcription file does not exist, path %s' % (id, path))
			return

		self.transcriptions = []
		with open(path, 'r', encoding = 'utf-8') as fin:
			for line in fin.readlines():
				words = line.strip().split(' ')

				# remove special symbol
				for element in self.black_list:
					while element in words:
						words.remove(element)

				self.transcriptions.append(words)

		scoring_dir = vote_dir + '/scoring_kaldi'

		if os.path.exists(scoring_dir) == False:
			os.mkdir(scoring_dir)

		path = scoring_dir + '/' + file_name

		self.transcription_file = open(path, 'a', encoding = 'utf-8')

		self.file_name = file_name
		self.id = id
		self.weight_function = weight_function
		self.backward = backward
		self.model = None
		self.tokenizer = None

		self.init = True

	def __del__(self):
		if self.init == False:
			return

		self.transcription_file.close()

	def get_uid_list(self):
		if self.init == False:
			logging.error('get_uid_list(%d): initialization not complete' % (self.id))
			return []

		uids = []
		for transcription in self.transcriptions:
			uids.append(transcription[0])
		return uids

	def set_model(self, model, tokenizer):
		if self.init == False:
			logging.error('set_model(%d): initialization not complete' % (self.id))
			return

		self.model = model
		self.tokenizer = tokenizer

		return

	# create voter object for specific uid
	def create_voter(self, uid):
		if self.init == False:
			logging.error('create_voter(%d): initialization not complete' % (self.id))
			return None

		for transcription in self.transcriptions:
			if uid != transcription[0]:
				continue

			voter = Voter(self.id, self.weight_function, self.backward)

			voter.init_by_list(transcription)
			voter.set_model(self.model, self.tokenizer)

			logging.debug('create_voter(%d): voter created with transcription %s' % (self.id, transcription))
			return voter

		logging.error('create_voter(%d): fail to find transcription for uid %s' % (self.id, uid))
		return None

	# print uid/filename and transcription to a file
	def print_transcription(self, uid, file, print_uid = True):
		if self.init == False:
			logging.error('print_transcription(%d): initialization not complete' % (self.id))
			return False

		for transcription in self.transcriptions:
			if uid != transcription[0]:
				continue

			if print_uid != False:
				file.write(transcription[0] + ' ')
			else:
				file.write(self.file_name + ' ')

			for i in range(1, len(transcription)):
				file.write(transcription[i] + ' ')

			file.write('\n')
			return True

		logging.error('print_transcription(%d): fail to find transcription for uid %s' % (self.id, uid))
		return False

def find_max_vote(votes, weights):
	max_weight = float('-inf')
	max_vote = ''
	for vote in votes:
		weight = weights[vote]
		if weight > max_weight:
			max_weight = weight
			max_vote = vote

	return max_vote

def do_vote(voters):
	current_votes = []
	all_weights = {}
	for voter in voters:
		all_votes = []

		used_votes, current_vote, future_votes = voter.do_vote()

		# used_votes, future_votes are lists while current_vote is string
		current_votes.append(current_vote)

		all_votes += used_votes
		all_votes += future_votes
		all_votes.append(current_vote)

		for vote in all_votes:
			if vote in all_weights:
				all_weights[vote] += voter.weight
			else:
				all_weights[vote] = voter.weight

	max_vote = find_max_vote(current_votes, all_weights)

	logging.info('do_vote: winner %s, weights %s' % (repr(max_vote), all_weights))
	return max_vote

def do_align(voters, winner):
	current_votes = []
	all_weights = {}
	for voter in voters:
		vote = voter.do_align(winner)

		# current_vote is string
		current_votes.append(vote)

		if vote in all_weights:
			all_weights[vote] += voter.weight
		else:
			all_weights[vote] = voter.weight

	max_vote = find_max_vote(current_votes, all_weights)

	logging.info('do_align: winner %s, weights %s' % (repr(max_vote), all_weights))

	if max_vote != winner:
		# it could happen, not sure if it's an error or not
		logging.warning('do_align: inconsistent result, winner %s, max_vote %s' % (repr(winner), repr(max_vote)))

	return True

def is_subseq(possible_subseq, seq):
	if len(possible_subseq) > len(seq):
		return False
	def get_length_n_slices(n):
		for i in range(len(seq) + 1 - n):
			yield seq[i:i+n]
	for slyce in get_length_n_slices(len(possible_subseq)):
		if slyce == possible_subseq:
			return True
	return False

def is_subseq_of_any(find, data):
	if len(data) < 1 and len(find) < 1:
		return False
	for i in range(len(data)):
		if not is_subseq(find, data[i]):
			return False
	return True

def get_longest_common_subseq(data):
	substr = []
	if len(data) > 1 and len(data[0]) > 0:
		for i in range(len(data[0])):
			for j in range(len(data[0])-i+1):
				if j > len(substr) and is_subseq_of_any(data[0][i:i+j], data):
					substr = data[0][i:i+j]
	return substr

def process_voters(voters, action):
	result = []

	if action == 'vote' or action == 'lcs':
		if action == 'lcs':
			data = []

			for voter in voters:
				data.append(voter.plain_text)

			lcs = get_longest_common_subseq(data)

			if len(lcs) != 0:
				logging.debug('process_voters: lcs %s found' % (lcs))

				# divide-and-conquer
				lchildren = []
				rchildren = []

				lempty = 0
				rempty = 0
				for voter in voters:
					idx = voter.plain_text.find(lcs)

					if idx < 0:
						# should not happen
						logging.error('process_voters: fail to find lcs \'%s\' in sentence \'%s\'' % (lcs, voter.plain_text))
						return []

					lsentence = voter.plain_text[:idx]
					lchild = voter.fork_voter(lsentence)
					if lchild == None:
						logging.error('process_voters: fail to fork voter with sentence \'%s\'' % (lsentence))
						return []

					lchildren.append(lchild)

					rsentence = voter.plain_text[idx + len(lcs):]
					rchild = voter.fork_voter(rsentence)
					if rchild == None:
						logging.error('process_voters: fail to fork voter with sentence \'%s\'' % (rsentence))
						return []

					rchildren.append(rchild)

					# lchild has nothing to vote
					if idx == 0:
						lempty += 1

					# rchild has nothing to vote
					if idx + len(lcs) >= len(voter.plain_text):
						rempty += 1

				if len(voters) != lempty:
					result += process_voters(lchildren, action)

				result.append(lcs)

				if len(voters) != rempty:
					result += process_voters(rchildren, action)

				return result

		# common part of both vote and lcs
		while (True):
			winner = do_vote(voters)
			if winner == None:
				logging.error('process_voters: fail to vote')
				return []

			if do_align(voters, winner) == False:
				logging.error('process_voters: fail to align')
				return []

			if winner == '\n':
				break

			result.append(winner)
	elif action == 'vote-combine' or action == 'lcs-combine':
		forward_text = voters[0].plain_text
		backward_text = voters[1].plain_text

		if forward_text == backward_text:
			# fast path
			max_voter = voters[0]
		else:
			for voter in voters:
				voter.update_weight()

			max_weight = float('-inf')
			for voter in voters:
				if (voter.weight > max_weight):
					max_weight = voter.weight
					max_voter = voter

		result = max_voter.remain_votes
	else:
		logging.error('process_voters: unknown action \'%s\'' % (action))

	return result

def process_hypothesis_files(vote_dir, files, weight_function, direction, action):
	backward = False
	factories = []
	id = 0

	if action == 'vote-combine' or action == 'lcs-combine':
		if len(files) != 2:
			logging.error('process_hypothesis_files: only process two vote files')
			return False

	if direction == 'backward':
		backward = True

	for file in files:
		factory = VoterFactory(vote_dir, file, id, weight_function, backward)
		if factory.init == False:
			logging.error('process_hypothesis_files: fail to create factory %d from file %s' % (id, file))
			return False

		factories.append(factory)
		id += 1

	uids = factories[0].get_uid_list()
	logging.info('process_hypothesis_files: %d uids from factory %d' % (len(uids), factories[0].id))

	scoring_dir = vote_dir + '/scoring_kaldi'

	if os.path.exists(scoring_dir) == False:
		os.mkdir(scoring_dir)

	fout = open(scoring_dir + '/vote.txt', 'w', encoding = 'utf-8')
	fout_all = open(scoring_dir + '/vote-all.txt', 'w', encoding = 'utf-8')

	truth = VoterFactory(vote_dir, 'test_filt.txt', id, weight_function, backward)
	if truth.init == False:
		logging.error('process_hypothesis_files: fail to create factory %d from file %s' % (id, 'test_filt.txt'))
		return False

	if weight_function == 'ppl' or weight_function == 'ppl-count':
		model = BertForMaskedLM.from_pretrained('hfl/chinese-bert-wwm-ext')
		model.eval()

		# Load pre-trained model tokenizer (vocabulary)
		tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm-ext')

		for factory in factories:
			factory.set_model(model, tokenizer)

	for uid in uids:
		logging.info('process_hypothesis_files: vote for uid %s' % (uid))

		# save to test_filt.txt in scoring directory
		if truth.print_transcription(uid, truth.transcription_file) == False:
			logging.error('process_hypothesis_files: fail to save transcription of ground truth')
			return False

		# save to vote-all.txt in scoring directory
		if truth.print_transcription(uid, fout_all) == False:
			logging.error('process_hypothesis_files: fail to print transcription of ground truth')
			return False

		voters = []
		for factory in factories:
			# save to file in scoring directory
			if factory.print_transcription(uid, factory.transcription_file) == False:
				logging.error('process_hypothesis_files: fail to save transcription of factory %d' % (factory.id))
				return False

			# save to vote-all.txt in scoring directory
			if factory.print_transcription(uid, fout_all, print_uid = False) == False:
				logging.error('process_hypothesis_files: fail to print transcription factory %d' % (factory.id))
				return False

			voter = factory.create_voter(uid)
			if voter == None:
				logging.error('process_hypothesis_files: fail to create voter for factory %d' % (factory.id))
				return False

			if action != 'vote-combine' and action != 'lcs-combine':
				voter.update_weight()

			voters.append(voter)

		result = process_voters(voters, action)

		logging.info('process_hypothesis_files: result %s' % (result))

		# save to vote.txt and vote-all.txt in scoring directory
		fout.write(uid + ' ')
		fout_all.write('vote ')

		if backward != False:
			result.reverse()
			for r in result:
				fout.write(r[::-1] + ' ')
				fout_all.write(r[::-1] + ' ')
		else:
			for r in result:
				fout.write(r + ' ')
				fout_all.write(r + ' ')

		fout.write('\n')
		fout_all.write('\n\n')

		#fout.flush()
		#fout_all.flush()

	fout.close()
	fout_all.close()

	return True

def parse_best_cer_file(decode_dir):
	best_cer = decode_dir + '/scoring_kaldi/best_cer'

	if os.path.exists(best_cer) == False:
		logging.error('best_cer file does not exist, path %s' % (best_cer))
		return None

	pattern = re.compile(r'(?<=/cer_)(\d+)_(\d+\.\d+)')

	with open(best_cer, 'r', encoding = 'utf-8') as fin:
		for line in fin.readlines():
			m = pattern.search(line)
			num = m.group(1)
			penalty = m.group(2)

			logging.debug('best_cer: line %s, num %s, penalty %s' % (repr(line), num, penalty))

			path = decode_dir + '/scoring_kaldi/penalty_' + penalty + '/' + num + '.txt'
			logging.info('best_cer: path %s' % (path))
			return path

	logging.error('fail to parse best_cer file')
	return None

def init_vote_data_directory(decode_root, vote_dir, test_set, lm_name, lm_tests, lm_subsets):
	if len(lm_tests) != len(lm_subsets):
		logging.error('lengh of lm_tests and lm_subsets do not match')
		return []

	# create data directory
	data_dir = vote_dir + '/data'
	os.mkdir(data_dir)

	# copy transcript file of ground truth to data directory
	shutil.copyfile('./data/' + test_set + '/test/text', data_dir + '/test_filt.txt')

	# copy/append hypothesis file(s) of each LM to data directory
	hypothesis_files = []
	for i in range(len(lm_tests)):
		if lm_subsets[i] != 1:
			for subset in range(1, lm_subsets[i] + 1):
				decode_dir = decode_root + '/decode_' + test_set + '-' + str(subset) + '_' + lm_name + '_tg_' + lm_tests[i]

				hypothesis_file = parse_best_cer_file(decode_dir)
				if hypothesis_file == None:
					logging.error('fail to find hypothesis file, decode dir %s' % (decode_dir))
					return []

				if os.path.exists(hypothesis_file) == False:
					logging.error('hypothesis file does not exist, path %s' % (hypothesis_file))
					return []

				dst = data_dir + '/' + lm_tests[i] + '.txt'
				with open(dst, 'a', encoding = 'utf-8') as fout:
					with open(hypothesis_file, 'r', encoding = 'utf-8') as fin:
						for line in fin.readlines():
							fout.write(line)
		else:
			decode_dir = decode_root + '/decode_' + test_set + '_' + lm_name + '_tg_' + lm_tests[i]

			hypothesis_file = parse_best_cer_file(decode_dir)
			if hypothesis_file == None:
				logging.error('fail to find hypothesis file, decode dir %s' % (decode_dir))
				return []

			if os.path.exists(hypothesis_file) == False:
				logging.error('hypothesis file does not exist, path %s' % (hypothesis_file))
				return []

			dst = data_dir + '/' + lm_tests[i] + '.txt'
			shutil.copyfile(hypothesis_file, dst)

		hypothesis_files.append(lm_tests[i] + '.txt')

	return hypothesis_files

def init_combine_data_directory(vote_dir_common, vote_dir):
	hypothesis_files = ['vote-forward.txt', 'vote-backward.txt']

	forward_vote_dir = vote_dir_common + '_forward'
	if os.path.exists(forward_vote_dir) == False:
		print('forward vote directory %s does not exist' % (forward_vote_dir))
		return []

	backward_vote_dir = vote_dir_common + '_backward'
	if os.path.exists(backward_vote_dir) == False:
		print('backward vote directory %s does not exist' % (backward_vote_dir))
		return []

	# create data directory
	data_dir = vote_dir + '/data'
	os.mkdir(data_dir)

	shutil.copyfile(forward_vote_dir + '/scoring_kaldi/vote.txt', data_dir + '/vote-forward.txt')
	shutil.copyfile(backward_vote_dir + '/scoring_kaldi/vote.txt', data_dir + '/vote-backward.txt')

	shutil.copyfile(forward_vote_dir + '/scoring_kaldi/test_filt.txt', data_dir + '/test_filt.txt')
	#shutil.copyfile(backward_vote_dir + '/scoring_kaldi/test_filt.txt', data_dir + '/test_filt-backward.txt')

	return hypothesis_files

def setup_logging(log_dir):
	log_path = log_dir + '/log.txt'

	print('setup_logging: log %s' % (log_path))

	# setup logging
	logging.basicConfig(level = logging.DEBUG,
			    format = '%(asctime)s - %(levelname)s - %(message)s',
			    filename = log_path,
			    filemode = 'w')
	# define a new Handler to log to console as well
	console = logging.StreamHandler()
	# optional, set the logging level
	console.setLevel(logging.INFO)
	# set a format which is the same for console use
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	# tell the handler to use this format
	console.setFormatter(formatter)
	# add the handler to the root logger
	logging.getLogger('').addHandler(console)

	return

def main():
	backward = False

	parser = argparse.ArgumentParser()

	parser.add_argument('--decode_root', type = str, default = './exp/chain_cleaned/tdnn_cnn_1a_sp', help = 'root directory of decode data')

	parser.add_argument('--test_set', type = str, default = 'mf_v3', help = 'name of test set')
	parser.add_argument('--lm_name', type = str, default = 'lm_v5_interp_lexicon', help = 'name of lm (with suffix)')

	parser.add_argument('--lm_tests', nargs = '+', type = str, default = 'pytorch_transformer_e0.05_w0.5 pytorch_transformer_e0.1_w0.5 rnnlm_1e_0.45 pytorch_lstm_e0.05_w0.5 pytorch_lstm_e0.1_w0.5', help = 'name of lm tests')
	parser.add_argument('--lm_subsets', nargs = '+', type = int, help = 'number of subsets of lm')

	# count, ppl, ppl-count
	parser.add_argument('--weight_function', type = str, default = 'count', help = 'weight function')
	parser.add_argument('--direction', type = str, default = 'forward', help = 'direction for voting')

	# vote, lcs, vote-combine, lcs-combine
	parser.add_argument('action', help = 'action to take')

	args = parser.parse_args()

	decode_root = args.decode_root
	test_set = args.test_set
	lm_name = args.lm_name
	lm_tests = args.lm_tests
	lm_subsets = args.lm_subsets
	weight_function = args.weight_function
	direction = args.direction

	action = args.action

	weight_function_options = ['count', 'ppl', 'ppl-count']
	direction_options = ['forward', 'backward']
	action_options = ['vote', 'lcs', 'vote-combine', 'lcs-combine']

	if weight_function not in weight_function_options:
		logging.error('unknown weight_function \'%s\'' % (weight_function))
		return

	if direction not in direction_options:
		logging.error('unknown direction \'%s\'' % (direction))
		return

	if action not in action_options:
		logging.error('unknown action \'%s\'' % (action))
		return

	if action == 'vote' or action == 'lcs':
		# create vote directory if not exist
		if action == 'vote':
			vote_dir = decode_root + '/vote_'
		else:
			vote_dir = decode_root + '/lcs_'
		vote_dir += test_set + '_' + lm_name + '_' + weight_function + '_' + direction

		if os.path.exists(vote_dir) != False:
			print('remove old vote directory %s' % (vote_dir))
			shutil.rmtree(vote_dir)

		os.mkdir(vote_dir)

		setup_logging(vote_dir)

		# copy/merge hypothesis files to data directory
		hypothesis_files = init_vote_data_directory(decode_root, vote_dir, test_set, lm_name, lm_tests, lm_subsets)
		if len(hypothesis_files) == 0:
			logging.error('fail to init data directory')
			return

		if process_hypothesis_files(vote_dir, hypothesis_files, weight_function, direction, action) == False:
			logging.error('fail to process hypothesis files')
			return
	elif action == 'vote-combine' or action == 'lcs-combine':
		if action == 'vote-combine':
			vote_dir_common = decode_root + '/vote_'
		elif action == 'lcs-combine':
			vote_dir_common = decode_root + '/lcs_'

		vote_dir_common += test_set + '_' + lm_name + '_' + weight_function

		vote_dir = vote_dir_common + '_combine'
		if os.path.exists(vote_dir) != False:
			logging.info('remove old combine vote directory %s' % (vote_dir))
			shutil.rmtree(vote_dir)

		os.mkdir(vote_dir)

		setup_logging(vote_dir)

		hypothesis_files = init_combine_data_directory(vote_dir_common, vote_dir)
		if len(hypothesis_files) == 0:
			logging.error('fail to init data directory')
			return

		if process_hypothesis_files(vote_dir, hypothesis_files, 'ppl', 'forward', action) == False:
			logging.error('fail to process hypothesis files')
			return
	else:
		logging.error('unknown action \'%s\'' % (action))

	return

if __name__ == '__main__':
	main()
