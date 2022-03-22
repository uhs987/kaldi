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
	def __init__(self, id, backward):
		self.id = id
		self.backward = backward
		self.model = None
		self.tokenizer = None

		self.init = False

	def init_by_list(self, transcription):
		# init by VoterFactory, always got forward text
		self.plain_text = ''.join(transcription[1:])

		self.weight = 1.0

		if self.backward != False:
			self.plain_text = self.plain_text[::-1]

			# reverse all words and characters inside the words
			transcription[1:] = transcription[:0:-1]
			for i in range(1, len(transcription)):
				transcription[i] = transcription[i][::-1]

		self.unmatch_votes = []
		self.remain_votes = []
		self.remain_votes = list(transcription[1:])

		self.init = True

		logging.debug('init_by_list(%d), votes %s' % (self.id, self.remain_votes))
		return

	def init_by_string(self, sentence):
		# init by other Voter, may be backward text
		self.plain_text = sentence

		#self.weight = 1.0

		self.unmatch_votes = []
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
		self.model = model
		self.tokenizer = tokenizer

		return

	def fork_voter(self, sentence):
		if self.init == False:
			logging.error('fork_voter(%d): initialization not complete' % (self.id))
			return None

		voter = Voter(self.id, self.backward)

		voter.init_by_string(sentence)
		voter.set_model(self.model, self.tokenizer)
		voter.weight = self.weight

		logging.debug('fork_voter(%d), sentence %s' % (self.id, sentence))
		return voter

	def calculate_ppl(self):
		if self.init == False:
			logging.error('calculate_ppl(%d): initialization not complete' % (self.id))
			return

		if self.model == None:
			logging.error('calculate_ppl(%d): no model available' % (self.id))
			return

		if self.tokenizer == None:
			logging.error('calculate_ppl(%d): no tokenizer available' % (self.id))
			return

		text = self.plain_text
		model = self.model
		tokenizer = self.tokenizer

		if self.backward != False:
			text = text[::-1]

		with torch.no_grad():
			tokenize_input = tokenizer.tokenize(text)
			tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
			sen_len = len(tokenize_input)

			if sen_len == 0:
				ppl = float("inf")
			else:
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
			self.weight = ppl
		return

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

		logging.info('do_vote(%d): current %s, unmatch %s, future %s' % (self.id, repr(self.current_vote), self.unmatch_votes, future_votes))
		return self.unmatch_votes, self.current_vote, future_votes

	def do_align(self, vote):
		if self.init == False:
			logging.error('do_align(%d): initialization not complete' % (self.id))
			return

		if vote == self.current_vote:
			logging.info('do_align(%d): my vote wins, clean the unmatch %s' % (self.id, self.unmatch_votes))
			self.unmatch_votes = []
			return vote

		if len(vote) > len(self.current_vote) and len(self.remain_votes) > 0:
			if vote == self.current_vote + self.remain_votes[0]:
				logging.info('do_align(%d): my vote %s should win but too short, pop next vote %s' % (self.id, repr(self.current_vote), repr(self.remain_votes[0])))
				self.remain_votes.pop(0)
				self.unmatch_votes = []
				return vote

		if len(vote) < len(self.current_vote):
			if vote == self.current_vote[:len(vote)]:
				remain = self.current_vote[len(vote):]
				self.remain_votes.insert(0, remain)
				self.unmatch_votes = []
				logging.info('do_align(%d): my vote %s should win but too long, trim it and update remain %s' % (self.id, repr(self.current_vote), self.remain_votes))
				return vote

		if vote in self.unmatch_votes:
			self.unmatch_votes.append(self.current_vote)

			# remove obsolete votes
			while self.unmatch_votes.pop(0) != vote:
				pass

			if len(self.unmatch_votes) != 0:
				# insert back to remain_votes for next do_vote() call
				self.unmatch_votes.extend(self.remain_votes)
				self.remain_votes = self.unmatch_votes
				self.unmatch_votes = []

			logging.info('do_align(%d): winner %s in unmatch, update remain %s' % (self.id, repr(vote), self.remain_votes))
			return vote

		# search if the vote is in remain_votes list
		found = False
		for i in range(min(2, len(self.remain_votes))):
			if self.remain_votes[i] == vote:
				found = True

		if found != False:
			# align myself
			while self.remain_votes.pop(0) != vote:
				pass

			self.unmatch_votes = []
			logging.info('do_align(%d): winner %s in remain, update remain %s' % (self.id, repr(vote), self.remain_votes))
			return vote

		self.unmatch_votes.append(self.current_vote)
		logging.info('do_align(%d): add %s to unmatch %s' % (self.id, repr(self.current_vote), self.unmatch_votes))

		#while len(self.unmatch_votes) > 4:
		#	self.unmatch_votes.pop(0)

		return self.current_vote

class VoterFactory:
	black_list = ['[SPK]', '[FIL]']

	def __init__(self, vote_dir, hypothesis_file, id, backward):
		self.init = False

		data_dir = vote_dir + '/data'

		path = data_dir + '/' + hypothesis_file

		if os.path.exists(path) == False:
			logging.error('VoterFactory(%d): hypothesis file does not exist, path %s' % (id, path))
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

		self.vote_dir = vote_dir
		self.hypothesis_file = hypothesis_file
		self.id = id
		self.backward = backward
		self.model = None
		self.tokenizer = None

		self.init = True

	def get_uid_list(self):
		if self.init == False:
			logging.error('get_uid_list(%d): initialization not complete' % (self.id))
			return

		uids = []
		for transcription in self.transcriptions:
			uids.append(transcription[0])
		return uids

	def get_transcription(self, uid):
		if self.init == False:
			logging.error('get_transcription(%d): initialization not complete' % (self.id))
			return

		for transcription in self.transcriptions:
			if uid == transcription[0]:
				return transcription

		logging.error('get_transcription(%d): fail to find hypothesis for uid %s' % (self.id, uid))
		return []

	def set_model(self, model, tokenizer):
		self.model = model
		self.tokenizer = tokenizer

		return

	def create_voter(self, uid):
		if self.init == False:
			logging.error('create_voter(%d): initialization not complete' % (self.id))
			return

		for transcription in self.transcriptions:
			if uid == transcription[0]:
				voter = Voter(self.id, self.backward)

				voter.init_by_list(transcription)
				voter.set_model(self.model, self.tokenizer)

				logging.debug('create_voter(%d), transcription %s' % (self.id, transcription))
				return voter

		logging.error('create_voter(%d): fail to find hypothesis for uid %s' % (self.id, uid))
		return None

	def save_hypothesis(self, uid):
		if self.init == False:
			logging.error('save_hypothesis(%d): initialization not complete' % (self.id))
			return

		cer_dir = self.vote_dir + '/scoring_kaldi'

		if os.path.exists(cer_dir) == False:
			os.mkdir(cer_dir)

		path = cer_dir + '/' + self.hypothesis_file

		with open(path, 'a', encoding = 'utf-8') as fout:
			for transcription in self.transcriptions:
				if uid != transcription[0]:
					continue

				for item in transcription:
					fout.write(item + ' ')

				fout.write('\n')
				return True

		logging.error('save_hypothesis(%d): fail to find hypothesis for uid %s' % (self.id, uid))
		return False

def do_vote(voters, weight):
	current_votes = []
	all_weights = {}
	for voter in voters:
		all_votes = []

		unmatch_votes, current_vote, future_votes = voter.do_vote()

		# unmatch_votes, future_votes are lists while current_vote is string
		current_votes.append(current_vote)

		all_votes += unmatch_votes
		all_votes += future_votes
		all_votes.append(current_vote)

		for vote in all_votes:
			if vote in all_weights:
				all_weights[vote] += (1.0 / voter.weight)
			else:
				all_weights[vote] = (1.0 / voter.weight)

			if weight == 'ppl-count':
				# consider ppl only when the word cound is the same
				all_weights[vote] += 1.0

	if len(current_votes) != len(voters):
		logging.error('do_vote: someone not voting')
		return None

	max_weight = float("-inf")
	max_vote = ''
	for vote in current_votes:
		weight = all_weights[vote]
		if (weight > max_weight):
			max_weight = weight
			max_vote = vote

	logging.info('do_vote: winner %s, weights %s' % (repr(max_vote), all_weights))
	return max_vote


def do_align(voters, weight, winner):
	current_votes = []
	all_weights = {}
	for voter in voters:
		vote = voter.do_align(winner)

		# current_vote is string
		current_votes.append(vote)

		if vote in all_weights:
			all_weights[vote] += (1.0 / voter.weight)
		else:
			all_weights[vote] = (1.0 / voter.weight)

		if weight == 'ppl-count':
			# consider ppl only when the word cound is the same
			all_weights[vote] += 1.0

	if len(current_votes) != len(voters):
		logging.error('do_align: someone not voting')
		return False

	max_weight = float("-inf")
	max_vote = ''
	for vote in current_votes:
		weight = all_weights[vote]
		if (weight > max_weight):
			max_weight = weight
			max_vote = vote

	logging.info('do_align: winner %s, weights %s' % (repr(max_vote), all_weights))

	if max_vote != winner:
		# it could happen, not sure if it's an error
		logging.warning("do_align: inconsistent result, winner %s, max_vote %s" % (repr(winner), repr(max_vote)))

	return True

def find_best_cer_hypothesis_file(decode_dir):
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

def init_data_directory(decode_root, vote_dir, test_set, lm_name, lm_tests, lm_subsets):
	if len(lm_tests) != len(lm_subsets):
		logging.error('lengh of lm_tests and lm_subsets do not match')
		return []

	# create data directory
	data_dir = vote_dir + '/data'
	os.mkdir(data_dir)

	# copy transcript file
	shutil.copyfile('./data/' + test_set + '/test/text', data_dir + '/test_filt.txt')

	# copy/append hypothesis file
	files = []
	for i in range(len(lm_tests)):
		if lm_subsets[i] != 1:
			for subset in range(1, lm_subsets[i] + 1):
				decode_dir = decode_root + '/decode_' + test_set + '-' + str(subset) + '_' + lm_name + '_tg_' + lm_tests[i]

				hypothesis_file = find_best_cer_hypothesis_file(decode_dir)
				if hypothesis_file == None:
					logging.error('fail to find hypothesis file, decode dir %s' % (decode_dir))

				if os.path.exists(hypothesis_file) == False:
					logging.error('hypothesis file does not exist, path %s' % (hypothesis_file))

				dst = data_dir + '/' + lm_tests[i] + '.txt'
				with open(dst, 'a', encoding = 'utf-8') as fout:
					with open(hypothesis_file, 'r', encoding = 'utf-8') as fin:
						for line in fin.readlines():
							fout.write(line)
		else:
			decode_dir = decode_root + '/decode_' + test_set + '_' + lm_name + '_tg_' + lm_tests[i]

			hypothesis_file = find_best_cer_hypothesis_file(decode_dir)
			if hypothesis_file == None:
				logging.error('fail to find hypothesis file, decode dir %s' % (decode_dir))

			if os.path.exists(hypothesis_file) == False:
				logging.error('hypothesis file does not exist, path %s' % (hypothesis_file))

			dst = data_dir + '/' + lm_tests[i] + '.txt'
			shutil.copyfile(hypothesis_file, dst)

		files.append(lm_tests[i] + '.txt')

	return files

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

def process_sentence(voters, weight, action):
	result = []

	if action == 'vote' or action == 'lcs':
		if action == 'lcs':
			data = []

			for voter in voters:
				data.append(voter.plain_text)

			lcs = get_longest_common_subseq(data)

			if len(lcs) != 0:
				logging.debug('process_sentence: lcs %s found' % (lcs))

				# divide-and-conquer
				lefters = []
				righters = []
				for voter in voters:
					idx = voter.plain_text.find(lcs)

					if idx != 0:
						child = voter.fork_voter(voter.plain_text[:idx])
						if child == None:
							logging.debug('process_sentence: fail to fork voter')
							return []

						lefters.append(child)

					if idx + len(lcs) < len(voter.plain_text):
						child = voter.fork_voter(voter.plain_text[idx + len(lcs):])
						if child == None:
							logging.debug('process_sentence: fail to fork voter')
							return []

						righters.append(child)

				if len(lefters) != 0:
					result += process_sentence(lefters, weight, action)

				result.append(lcs)

				if len(righters) != 0:
					result += process_sentence(righters, weight, action)

				return result

		while (True):
			winner = do_vote(voters, weight)
			if winner == None:
				logging.error("fail to vote")
				return False

			if do_align(voters, weight, winner) == False:
				logging.error("fail to align")
				return False

			if winner == '\n':
				break

			result.append(winner)
	elif action == 'vote-combine' or action == 'lcs-combine':
		forward_text = voters[0].plain_text
		backward_text = voters[1].plain_text

		if forward_text == backward_text:
			# fast path
			min_voter = voters[0]
		else:
			for voter in voters:
				voter.calculate_ppl()

			min_weight = float("inf")
			for voter in voters:
				# weight is ppl, smaller is better
				if (voter.weight < min_weight):
					min_weight = voter.weight
					min_voter = voter

		result = min_voter.remain_votes

	return result

def process_hypothesis_files(vote_dir, files, weight, direction, action):
	backward = False
	factories = []
	id = 0

	if action == 'vote-combine' or action == 'lcs-combine':
		if len(files) != 2:
			logging.error('only process two vote files')
			return False

	if direction == 'backward':
		backward = True

	for file in files:
		factory = VoterFactory(vote_dir, file, id, backward)
		if factory.init == False:
			logging.error('fail to create factory %d from file %s' % (id, file))
			return False

		factories.append(factory)
		id += 1

	uids = factories[0].get_uid_list()
	logging.info('process_hypothesis_files: %d uids from factory %d' % (len(uids), factories[0].id))

	cer_dir = vote_dir + '/scoring_kaldi'

	if os.path.exists(cer_dir) == False:
		os.mkdir(cer_dir)

	fout = open(cer_dir + '/vote.txt', 'w', encoding = 'utf-8')
	fout_all = open(cer_dir + '/vote-all.txt', 'w', encoding = 'utf-8')

	answer = VoterFactory(vote_dir, 'test_filt.txt', id, backward)
	if answer.init == False:
		logging.error('fail to create factory %d from file %s' % (id, 'test_filt.txt'))
		return False

	if weight == 'ppl' or weight == 'ppl-count':
		model = BertForMaskedLM.from_pretrained('hfl/chinese-bert-wwm-ext')
		model.eval()

		# Load pre-trained model tokenizer (vocabulary)
		tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm-ext')

		for factory in factories:
			factory.set_model(model, tokenizer)

	for uid in uids:
		logging.info('process_hypothesis_files: vote for uid %s' % (uid))

		# save for later cer computation
		answer.save_hypothesis(uid)

		words = answer.get_transcription(uid)
		for word in words:
			fout_all.write(word + ' ')
		fout_all.write('\n')

		voters = []
		for factory in factories:
			# save for later cer computation
			factory.save_hypothesis(uid)

			fout_all.write(factory.hypothesis_file + ' ')
			words = factory.get_transcription(uid)
			for i in range(1, len(words)):
				fout_all.write(words[i] + ' ')
			fout_all.write('\n')

			voter = factory.create_voter(uid)
			if voter == None:
				logging.error('factory %d fail to create voter' % (factory.id))
				break

			voters.append(voter)

		fout.write(uid + ' ')
		if backward != False:
			fout_all.write('vote(reversed) ')
		else:
			fout_all.write('vote ')

		if action != 'vote-combine' and action != 'lcs-combine':
			if weight == 'ppl' or weight == 'ppl-count':
				for voter in voters:
					voter.calculate_ppl()

		result = process_sentence(voters, weight, action)

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

		fout.flush()
		fout_all.flush()

		logging.info('process_hypothesis_files: result %s' % (result))

	fout.close()
	fout_all.close()

	return True

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
	parser.add_argument('--weight', type = str, default = 'count', help = 'weight function')
	parser.add_argument('--direction', type = str, default = 'forward', help = 'direction for voting')

	# vote, lcs, vote-combine, lcs-combine
	parser.add_argument("action", help = "action to take")

	args = parser.parse_args()

	decode_root = args.decode_root
	test_set = args.test_set
	lm_name = args.lm_name
	lm_tests = args.lm_tests
	lm_subsets = args.lm_subsets
	weight = args.weight
	direction = args.direction

	action = args.action

	if action == 'vote' or action == 'lcs':
		# create vote directory if not exist
		if action == 'vote':
			vote_dir = decode_root + '/vote_'
		else:
			vote_dir = decode_root + '/lcs_'
		vote_dir += test_set + '_' + lm_name + '_' + weight + '_' + direction

		if os.path.exists(vote_dir) != False:
			print('remove old vote directory %s' % (vote_dir))
			shutil.rmtree(vote_dir)

		os.mkdir(vote_dir)

		setup_logging(vote_dir)

		# copy/merge hypothesis files to data directory
		files = init_data_directory(decode_root, vote_dir, test_set, lm_name, lm_tests, lm_subsets)
		if len(files) == 0:
			logging.error('fail to init data directory')
			return

		if process_hypothesis_files(vote_dir, files, weight, direction, action) == False:
			logging.error('fail to process hypothesis files')
			return
	elif action == 'vote-combine' or action == 'lcs-combine':
		if action == 'vote-combine':
			vote_dir_common = decode_root + '/vote_'
		elif action == 'lcs-combine':
			vote_dir_common = decode_root + '/lcs_'

		vote_dir_common += test_set + '_' + lm_name + '_' + weight

		forward_vote_dir = vote_dir_common + '_forward'
		if os.path.exists(forward_vote_dir) == False:
			print('forward vote directory %s does not exist' % (forward_vote_dir))
			return

		backward_vote_dir = vote_dir_common + '_backward'
		if os.path.exists(backward_vote_dir) == False:
			print('backward vote directory %s does not exist' % (backward_vote_dir))
			return

		combine_vote_dir = vote_dir_common + '_combine'
		if os.path.exists(combine_vote_dir) != False:
			logging.info('remove old combine vote directory %s' % (combine_vote_dir))
			shutil.rmtree(combine_vote_dir)

		os.mkdir(combine_vote_dir)

		setup_logging(combine_vote_dir)

		# create data directory
		data_dir = combine_vote_dir + '/data'
		os.mkdir(data_dir)

		shutil.copyfile(forward_vote_dir + '/scoring_kaldi/vote.txt', data_dir + '/vote-forward.txt')
		shutil.copyfile(backward_vote_dir + '/scoring_kaldi/vote.txt', data_dir + '/vote-backward.txt')

		shutil.copyfile(forward_vote_dir + '/scoring_kaldi/test_filt.txt', data_dir + '/test_filt.txt')
		shutil.copyfile(backward_vote_dir + '/scoring_kaldi/test_filt.txt', data_dir + '/test_filt-backward.txt')

		files = ['vote-forward.txt', 'vote-backward.txt']
		if process_hypothesis_files(combine_vote_dir, files, 'ppl', 'forward', action) == False:
			logging.error('fail to process hypothesis files')
			return

	return

if __name__ == '__main__':
	main()
