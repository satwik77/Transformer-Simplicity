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

	parser.add_argument('-run_name', type=str, default='debug', help='run name for logs')
	parser.add_argument('-dataset', type=str, default='sparity40_5k', help='Dataset')

	parser.add_argument('-itr', dest='itr', action='store_true', help='Iteratively train')
	parser.add_argument('-no-itr', dest='itr', action='store_false', help='Train epochwise on fixed dataset')
	parser.set_defaults(itr=False)

	# Device Configuration
	parser.add_argument('-gpu', type=int, default=0, help='Specify the gpu to use')
	parser.add_argument('-seed', type=int, default=1729, help='Default seed to set')
	parser.add_argument('-logging', type=int, default=1, help='Set to 0 if you do not require logging')
	parser.add_argument('-ckpt', type=str, default='model', help='Checkpoint file name')
	

	
	parser.add_argument('-model_type', type=str, default='SAN', choices= ['SAN'],  help='Model Type: Transformer')
	
	parser.add_argument('-depth', type=int, default=2, help='Number of layers in each encoder and decoder')
	parser.add_argument('-dropout', type=float, default=0.1, help= 'Dropout probability for input/output/state units (0.0: no dropout)')

	parser.add_argument('-d_model', type=int, default=64, help='Embedding size in Transformer')
	parser.add_argument('-d_ffn', type=int, default=64, help='Hidden size of FFN in Transformer')
	parser.add_argument('-heads', type=int, default=4, help='Number of Attention heads in each layer')
	parser.add_argument('-pos_encode', default='learnable', choices= ['absolute','learnable'], help='Type of position encodings')
	parser.add_argument('-mask', dest='mask', action='store_true', help='Pos Mask')
	parser.add_argument('-no-mask', dest='mask', action='store_false', help='Do not Pos Mask')
	parser.set_defaults(mask=False)



	parser.add_argument('-init_range', type=float, default=0.08, help='Initialization range for seq2seq model')





	# Training parameters
	parser.add_argument('-lr', type=float, default=0.001, help='Learning rate')
	parser.add_argument('-decay_patience', type=int, default=3, help='Wait before decaying learning rate')
	parser.add_argument('-decay_rate', type=float, default=0.2, help='Amount by which to decay learning rate on plateu')
	parser.add_argument('-max_grad_norm', type=float, default=0.25, help='Clip gradients to this norm')
	parser.add_argument('-batch_size', type=int, default=500, help='Batch size')
	parser.add_argument('-epochs', type=int, default=1500, help='Maximum # of training epochs')
	parser.add_argument('-iters', type=int, default=40000, help='Maximum # of training iterations in iter mode')
	parser.add_argument('-opt', type=str, default='adam', choices=['adam', 'adadelta', 'sgd', 'asgd'], help='Optimizer for training')


	# Wandb
	parser.add_argument('-project', type=str, default='Bool', help='wandb project name')
	parser.add_argument('-entity', type=str, default='your_entity', help='wandb entity name')



	return parser