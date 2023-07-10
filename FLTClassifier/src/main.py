import os
import sys
import math
import logging
import pdb
import random
import numpy as np
# from attrdict import AttrDict
import torch
from torch.utils.data import DataLoader
# from collections import OrderedDict
try:
	import cPickle as pickle
except ImportError:
	import pickle

import wandb

from src.args import build_parser
from src.dataloader import Corpus, Sampler
from src.utils.helper import Voc, gpu_init_pytorch, create_save_directories, get_latest_checkpoint, count_parameters
from src.utils.logger import get_logger
from src.model import build_model, train_model


global log_folder
global model_folder
global result_folder
global data_path

log_folder = 'logs'
model_folder = 'models'
result_folder = './out/'
data_path = './data/'
# board_path = './runs/'


def load_data(config, logger):
	'''
		Loads the data from the datapath in torch dataset form

		Args:
			config (dict) : configuration/args
			logger (logger) : logger object for logging

		Returns:
			dataloader(s) 
	'''
	if config.mode == 'train':
		logger.debug('Loading Training Data...')

		'''Create Vocab'''
		train_path = os.path.join(data_path, config.dataset, 'train.pkl')
		val_path = os.path.join(data_path, config.dataset, 'dev.pkl')
		# test_path = os.path.join(data_path, config.dataset, 'test.tsv')
		voc= Voc()
		voc.create_vocab_dict(config, path= train_path, debug = config.debug)
		# voc.create_vocab_dict(config, path= val_path, debug = config.debug)
		# voc.create_vocab_dict(config, path= test_path, debug = config.debug)


		'''Load Datasets'''

		train_corpus = Corpus(train_path, voc, debug = config.debug)
		val_corpus = Corpus(val_path, voc, debug = config.debug)
		
		train_loader = Sampler(train_corpus, voc, config.batch_size)
		val_loader = Sampler(val_corpus, voc, config.batch_size)

		msg = 'Training and Validation Data Loaded:\nTrain Size: {}\nVal Size: {}'.format(len(train_corpus.data), len(val_corpus.data))
		logger.info(msg)


		return voc, train_loader, val_loader


	else:
		logger.critical('Invalid Mode Specified')
		raise Exception('{} is not a valid mode'.format(config.mode))



def main():
	'''read arguments'''

	print('Starting....')
	parser = build_parser()
	args = parser.parse_args()
	config = args
	mode= config.mode

	if mode == 'train':
		is_train = True
	else:
		is_train= False
	
	''' Set seed for reproducibility'''
	np.random.seed(config.seed)
	torch.manual_seed(config.seed)
	random.seed(config.seed)

	'''GPU initialization'''
	device = gpu_init_pytorch(config.gpu)

	
	

	'''Run Config files/paths'''
	run_name = config.run_name
	config.log_path = os.path.join(log_folder, run_name)
	config.model_path = os.path.join(model_folder, config.dataset, run_name)

	vocab_path = os.path.join(config.model_path, 'vocab.p')
	config_file = os.path.join(config.model_path, 'config.p')
	log_file = os.path.join(config.log_path, 'log.txt')

	if config.results:
		config.result_path = os.path.join(result_folder, 'val_results_{}.json'.format(config.dataset))
	

	if is_train:
		create_save_directories(config.log_path, config.model_path, result_folder)
	else:
		create_save_directories(config.log_path, None, result_folder)
	
	logger = get_logger(run_name, log_file, logging.DEBUG)

	logger.debug('Created Relevant Directories')
	logger.info('Experiment Name: {}'.format(config.run_name))

	
	if is_train:

		# logger.debug('Creating Vocab and loading Data ...')
		voc, train_loader, val_loader = load_data(config, logger)

		config.nlabels= train_loader.corpus.nlabels

		logger.info('Vocab Created with number of words : {}'.format(voc.nwords))		

		with open(vocab_path, 'wb') as f:
			pickle.dump(voc, f, protocol=pickle.HIGHEST_PROTOCOL)
	

	else:
		test_dataloader = load_data(config, logger)
		# logger.info('Loading Vocab File...')

		with open(vocab_path, 'rb') as f:
			voc = pickle.load(f)

		logger.info('Vocab Files loaded from {}'.format(vocab_path))

	
	checkpoint = get_latest_checkpoint(config.model_path, logger)

	if is_train:
		
		min_val_loss = torch.tensor(float('inf')).item()
		# min_val_ppl = float('inf')
		epoch_offset= 0


		if checkpoint:
			ckpt = torch.load(checkpoint, map_location=lambda storage, loc: storage)
			config.lr = ckpt['lr']
			model = build_model(config=config, voc=voc, device=device, logger=logger)
			model.load_state_dict(ckpt['model_state_dict'])
			model.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
		else:
			model = build_model(config=config, voc=voc, device=device, logger=logger)

		

		logger.info('Initialized Model')

		with open(config_file, 'wb') as f:
			pickle.dump(vars(config), f, protocol=pickle.HIGHEST_PROTOCOL)

		logger.debug('Config File Saved')


		num_params =count_parameters(model)
		logger.info('Number of parameters {}'.format(num_params))

		### Wandb Initialization ###
		if config.wandb:
			metrics = dict(
				num_params= num_params,
				learning_rate = config.lr,
				hidden_size= config.hidden_size,
				depth= config.depth,
				emb_size = config.emb_size,
				dataset= config.dataset,
				dropout= config.dropout,
				optimizer =config.opt,
				epochs = config.epochs,
				batch_size= config.batch_size,
			)
			
			wandb.init(
				project= config.project,
				group= config.dataset,
				name= config.run_name,
				config= metrics,
				entity=config.entity,
			)
		



		logger.info('Starting Training Procedure')
		train_model(model, train_loader, val_loader, voc,
					device, config, logger, epoch_offset, min_val_loss)






if __name__ == '__main__':
	main()