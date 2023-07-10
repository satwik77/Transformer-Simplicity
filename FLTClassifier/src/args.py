import argparse

### Add Early Stopping ###

def build_parser():
	# Data loading parameters
	parser = argparse.ArgumentParser(description='Run Transformer Classifier')


	# Mode specifications
	parser.add_argument('-mode', type=str, default='train', choices=['train', 'test'], help='Modes: train, test')
	# parser.add_argument('-debug', action='store_true', help='Operate on debug mode')
	parser.add_argument('-debug', dest='debug', action='store_true', help='Operate in debug mode')
	parser.add_argument('-no-debug', dest='debug', action='store_false', help='Operate in normal mode')
	parser.set_defaults(debug=False)
	parser.add_argument('-results', dest='results', action='store_true', help='Store results')
	parser.add_argument('-no-results', dest='results', action='store_false', help='Do not store results')
	parser.set_defaults(results=True)
	parser.add_argument('-wandb', dest='wandb', action='store_true', help='Store wandb')
	parser.add_argument('-no-wandb', dest='wandb', action='store_false', help='Do not store wandb')
	parser.set_defaults(wandb=False)
	parser.add_argument('-savei', dest='savei', action='store_true', help='save models in intermediate epochs')
	parser.add_argument('-no-savei', dest='savei', action='store_false', help='Do not save models in intermediate epochs')
	parser.set_defaults(savei=False)
	parser.add_argument('-init_dist', type=int, default=1, help='Log distance from init after x epochs. 0 indicates do not save')

	# Run name should just be alphabetical word (no special characters to be included)
	parser.add_argument('-run_name', type=str, default='debug', help='run name for logs')
	# parser.add_argument('-display_freq', type=int, default=35, help='number of batches after which to display loss')
	parser.add_argument('-dataset', type=str, default='sparity40_5k', help='Dataset')



	# Device Configuration
	parser.add_argument('-gpu', type=int, default=0, help='Specify the gpu to use')
	parser.add_argument('-seed', type=int, default=1729, help='Default seed to set')
	parser.add_argument('-logging', type=int, default=1, help='Set to 0 if you do not require logging')
	parser.add_argument('-ckpt', type=str, default='model', help='Checkpoint file name')
	

	# Dont modify ckpt_file
	# If you really want to then assign it a name like abc_0.pth.tar (You may only modify the abc part and don't fill in any special symbol. Only alphabets allowed
	# parser.add_argument('-date_fmt', type=str, default='%Y-%m-%d-%H:%M:%S', help='Format of the date')


	
	parser.add_argument('-model_type', type=str, default='RNN', choices= ['RNN'],  help='Model Type: Transformer')
	parser.add_argument('-cell_type', type=str, default='LSTM', choices= ['LSTM', 'GRU', 'RNN'],  help='RNN cell type, default: lstm')

	
	parser.add_argument('-depth', type=int, default=2, help='Number of layers in each encoder and decoder')
	parser.add_argument('-dropout', type=float, default=0.05, help= 'Dropout probability for input/output/state units (0.0: no dropout)')

	parser.add_argument('-emb_size', type=int, default=128, help='Embedding dimensions of inputs')
	parser.add_argument('-hidden_size', type=int, default=128, help='Number of hidden units in each layer')
	parser.add_argument('-tied', dest='tied', action='store_true', help='Tied Weights in input and output embeddings')
	parser.add_argument('-no-tied', dest='tied', action='store_false', help='Without Tied Weights in input and output embeddings')
	parser.set_defaults(tied=False)


	parser.add_argument('-init_range', type=float, default=0.08, help='Initialization range for seq2seq model')





	# Training parameters
	parser.add_argument('-lr', type=float, default=0.001, help='Learning rate')
	parser.add_argument('-decay_patience', type=int, default=3, help='Wait before decaying learning rate')
	parser.add_argument('-decay_rate', type=float, default=0.2, help='Amount by which to decay learning rate on plateu')
	parser.add_argument('-max_grad_norm', type=float, default=0.25, help='Clip gradients to this norm')
	parser.add_argument('-batch_size', type=int, default=32, help='Batch size')
	parser.add_argument('-epochs', type=int, default=1000, help='Maximum # of training epochs')
	parser.add_argument('-opt', type=str, default='adam', choices=['adam', 'adadelta', 'sgd', 'asgd'], help='Optimizer for training')
	

	# Wandb
	parser.add_argument('-project', type=str, default='Bool', help='wandb project name')
	parser.add_argument('-entity', type=str, default='your_entity', help='wandb entity name')


	return parser