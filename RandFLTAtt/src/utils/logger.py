import pdb
import logging
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
import json


def get_logger(name, log_file_path='./logs/temp.log', logging_level=logging.INFO):
	# log_format='%(asctime)s | %(levelname)s | %(filename)s: %(lineno)s : %(funcName)s() ::\t %(message)s'
	formatter= logging.Formatter('%(asctime)s | %(filename)s : %(funcName)s() ::\t %(message)s', "%m-%d %H")
	logger = logging.getLogger(name)
	logger.setLevel(logging_level)
	# formatter = logging.Formatter(log_format)

	file_handler = logging.FileHandler(log_file_path, mode='w')
	file_handler.setLevel(logging_level)
	file_handler.setFormatter(formatter)

	stream_handler = logging.StreamHandler()
	stream_handler.setLevel(logging_level)
	stream_handler.setFormatter(formatter)

	logger.addHandler(file_handler)
	logger.addHandler(stream_handler)

	# logger.addFilter(ContextFilter(expt_name))

	return logger


def print_log(logger, dict):
	string = ''
	for key, value in dict.items():
		string += '\n {}: {}\t'.format(key.replace('_', ' '), value)
	# string = string.strip()
	logger.info(string)


def store_results(config, val_score, train_acc, best_epoch):
	try:
		with open(config.result_path) as f:
			res_data =json.load(f)
	except:
		res_data = {}

	data= {'run_name' : config.run_name
	, 'val_score' : val_score
	, 'train_acc' : train_acc
	, 'best_epoch' : best_epoch
	, 'dataset' : config.dataset
	, 'heads': config.heads
	, 'd_model' : config.d_model
	, 'depth' : config.depth
	, 'dropout' : config.dropout
	, 'lr' : config.lr
	, 'batch_size' : config.batch_size
	, 'epochs' : config.epochs
	, 'opt' : config.opt
	}
	# res_data.update(data)
	res_data[str(config.run_name)] = data

	with open(config.result_path, 'w', encoding='utf-8') as f:
		json.dump(res_data, f, ensure_ascii= False, indent= 4)

