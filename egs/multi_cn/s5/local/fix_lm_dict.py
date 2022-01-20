#!/usr/bin/env python3

import argparse
import os
import shutil

root_dir='data/local/dict'

def main():

	# parse argument
	parser = argparse.ArgumentParser()
	parser.add_argument("lm_name", help="LM name")

	args = parser.parse_args()

	if (args.lm_name == None):
		return

	if os.path.isdir(root_dir+'/'+args.lm_name) == False:
		print('%s does not exist' % (root_dir+'/'+args.lm_name))
		return

	# backup files
	shutil.copy(root_dir+'/'+args.lm_name+'/lexicon.txt', root_dir+'/'+args.lm_name+'/lexicon.txt.bak')
	shutil.copy(root_dir+'/'+args.lm_name+'/nonsilence_phones.txt', root_dir+'/'+args.lm_name+'/nonsilence_phones.txt.bak')
	shutil.copy(root_dir+'/'+args.lm_name+'/extra_questions.txt', root_dir+'/'+args.lm_name+'/extra_questions.txt.bak')

	shutil.copy(root_dir+'/'+'/nonsilence_phones.txt', root_dir+'/'+args.lm_name+'/nonsilence_phones.txt')
	shutil.copy(root_dir+'/'+'/extra_questions.txt', root_dir+'/'+args.lm_name+'/extra_questions.txt')

	old_phones = set()
	with open(root_dir+'/'+'/nonsilence_phones.txt', 'r', encoding = 'UTF-8') as fin:
		for line in fin.readlines():
			old_phones.update(line.split())

	invalid_phones = set()
	with open(root_dir+'/'+args.lm_name+'/nonsilence_phones.txt.bak', 'r', encoding = 'UTF-8') as fin:
		for line in fin.readlines():
			invalid_phones.update(line.split())

	invalid_phones.difference_update(old_phones)
	#print(invalid_phones)

	os.remove(root_dir+'/'+args.lm_name+'/lexicon.txt')

	with open(root_dir+'/'+args.lm_name+'/lexicon.txt.bak', 'r', encoding = 'UTF-8') as fin:
		with open(root_dir+'/'+args.lm_name+'/lexicon.txt', 'w', encoding = 'UTF-8') as fout:

			for line in fin.readlines():
				x = line.split()
				found = False
				for i in range(1, len(x)):
					if x[i] in invalid_phones:
						found = True
						break
				if found == False:
					fout.write(line)

if __name__ == '__main__':
	main()
