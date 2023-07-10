import torch 
from torch import nn
import numpy as np
import ipdb as pdb
import random
import pandas as pd
from time import time
from attrdict import AttrDict
import argparse
import wandb

from src.components.rnnplay import RNNModel

from src.other_utils import sample_bstr, flip_i, Vocab, sents_to_idx, allbins

out_file= 'out/sensi_uni_exp.csv'
out_grouped= 'out/sensi_uni_group.csv'
out_model= './models/'

parser = argparse.ArgumentParser(description='')
parser.add_argument('-gpu', type=int, default=0, help='Specify the gpu to use')
# parser.add_argument('-run_name', type=str, default='SAN_sparseparity4a_h32_d2_exp1', help='run name for picking model')
parser.add_argument('-run', type=str, default='trialA', help='run name')
parser.add_argument('-sample_size', type=int, default=700, help='Sample size for estimation')

parser.add_argument('-len', type=int, default=20, help='Length of inputs')
parser.add_argument('-w_init', type=float, default=10.0, help='Range of weight values [-B,+B]')
parser.add_argument('-i_init', type=float, default=10.0, help='Range of input values [-B,+B]')
parser.add_argument('-hidden_size', type=int, default=64, help='Rnn hidden_size')
parser.add_argument('-depth', type=int, default=2, help='Rnn layers')
parser.add_argument('-model_type', type=str, default='LSTM', choices= ['RNN', 'LSTM', 'GRU'],  help='Model Type')

parser.add_argument('-trials', type=int, default=4, help='Number of trials')

parser.add_argument('-bias', dest='bias', action='store_true', help='Store bias')
parser.add_argument('-no-bias', dest='bias', action='store_false', help='Do not store bias')
parser.set_defaults(bias=False)


parser.add_argument('-wandb', dest='wandb', action='store_true', help='Store wandb')
parser.add_argument('-no-wandb', dest='wandb', action='store_false', help='Do not store wandb')
parser.set_defaults(wandb=False)

args = parser.parse_args()


run_name = args.run + '_'+ 'h{}d{}t{}m{}'.format(args.hidden_size, args.depth, args.trials, args.model_type) + '_'+ str(np.random.choice(100000))

if args.gpu >=0:
	device =torch.device('cuda:{}'.format(args.gpu))
else:
	device = torch.device('cpu')


grp = 'Uniform_len'+ str(args.len)

if args.wandb:
    metrics = dict(
        depth= args.depth,
        hidden_size = args.hidden_size,
        length= args.len,
        bias= args.bias,
        sample_size= args.sample_size,
        trials = args.trials,
        weight_range= args.w_init,
        model_type= args.model_type,
    )

    wandb.init(
        project= 'RandSensi-RNN',
        name= run_name,
        config= metrics,
        group= grp,
    )


def test_model(str_list, clf):
	clf.eval()
	wlens = torch.tensor([len(x) for x in str_list])
	batch_ids = sents_to_idx(voc, str_list)
	str_ids = batch_ids[:,:-1].transpose(0,1)
	str_ids = str_ids.to(device)
	wlens= wlens.to(device)
	
	with torch.no_grad():
		hidden = clf.init_hidden(len(str_list))
		output, hidden = clf(str_ids, hidden, wlens)
		preds = output.cpu().numpy()
		preds= preds.reshape(-1)
		preds = np.array(preds>=0.5, dtype=int)
		# preds = preds.argmax(axis=1)
		
	return preds
	



### Model Def #####
print('Loading Model')
rnn_type= args.model_type
nonlinearity='tanh'
weight_init= args.w_init
hidden_size = args.hidden_size
depth = args.depth
inp_init = args.i_init
inp_len = args.len
bias = args.bias

clf = RNNModel(ntoken =3, noutputs=1, nemb= args.hidden_size, nhid=args.hidden_size, nlayers= args.depth, rnn_type=rnn_type, nonlinearity=nonlinearity)
clf = clf.to(device)
voc = Vocab()





### Sensitivity Experiment ####

sample_size = args.sample_size
check_idx = list(range(inp_len))


### Pandas Data setup

cols = ['run_name', 'sample_size', 'len', 'rnn_type', 'hidden_size', 'depth', 'is_bias', 'weight_init', 'inp_init', 'trial', 'sensi_count', 'avg_sensi']
df= pd.DataFrame(data=[], columns=cols)
fixed_data= [run_name, sample_size, inp_len, rnn_type, hidden_size, depth, bias, weight_init, inp_init]

cols_group = ['run_name', 'sample_size', 'num_trials', 'len', 'rnn_type', 'hidden_size', 'depth', 'is_bias', 'weight_init', 'inp_init', 'avg_sensi', 'std_sensi']
big_df = pd.DataFrame(data=[], columns=cols_group)
group_data= [run_name, sample_size, args.trials, inp_len, rnn_type, hidden_size, depth, bias, weight_init, inp_init]






print('Generating Samples')
if args.len > 12:
	sample_points  =sample_bstr(sample_size, length= inp_len)
else:
	sample_points = allbins(args.len)
	sample_size = len(sample_points)


avg_list= []


for trial in range(args.trials):
	print('Running for trial : {}'.format(trial))
	start_time = time()
	clf.init_weights(weight_init= weight_init, inp_init= inp_init, dec_init= 1.0, bias= bias)   # inp_init and dec_init have no impact; Change def init_weights() in rnnplay.py if you want to use them

	sensi_count = 0
	avg_sensi = 0.0

	sample_preds = []
	for j in range(sample_size):
		bstr= sample_points[j]
		pred_bstr= test_model([bstr], clf)
		sample_preds.append(pred_bstr[0])

	assert sample_size == len(sample_preds)

	for i in check_idx:
		flag = 0
		mis_counter = 0
		# print('Testing for idx {}'.format(i))

		for j in range(sample_size):
			bstr = sample_points[j]
			fstr = flip_i(bstr, i)
			test_str = [fstr]

			preds = test_model(test_str, clf)
			pred_fstr= preds[0]
			pred_bstr = sample_preds[j]

			if pred_fstr != pred_bstr:
				if flag==0:
					# print('Idx {} sensitive after {} samples'.format(i, j))
					sensi_count+=1
					flag =1
				mis_counter+=1
		
		ratio = mis_counter/sample_size
		avg_sensi += ratio

	avg_sensi = avg_sensi/inp_len
	avg_list.append(avg_sensi)

	time_taken = time() - start_time
	time_mins = int(time_taken/60)
	time_secs= time_taken%60

	print('Trial {} completed with {} samples...\nTime Taken: {} mins and {} secs'.format(trial, sample_size, time_mins, time_secs))

	print('\n-------------Model {} Done--------------'.format(trial))
	print(trial, 'Done')
	print(sensi_count)
	print(avg_sensi)


	if avg_sensi>0.98:
		model_path = out_model+run_name+'_'+str(trial)+'.pt'
		torch.save(clf.state_dict(), model_path)
		print('Model Saved')


	if args.wandb:
		wandb.log({
			'Avg_sensi': avg_sensi,
			'Sensi_count': sensi_count,
		}, step= trial)

	new_data= fixed_data.copy() + [trial, sensi_count, avg_sensi]
	df.loc[trial] = new_data

avg_list = np.array(avg_list)
mean_sensi = avg_list.mean()
std_sensi = avg_list.std()
print('Mean Sensi: {}'.format(mean_sensi))




try:
	out_df = pd.read_csv(out_file)
	new_df = pd.concat([out_df, df], ignore_index=True)
	new_df.to_csv(out_file, index=False)

except:
	df.to_csv(out_file, index=False)


big_df.loc[0] = group_data + [mean_sensi, std_sensi]

try:
	out_df = pd.read_csv(out_grouped)
	new_df = pd.concat([out_df, big_df], ignore_index=True)
	new_df.to_csv(out_grouped, index=False)

except:
	big_df.to_csv(out_grouped, index=False)




if args.wandb:
	wandb.log({
		'mean_sensi': mean_sensi,
		'mean+std' : mean_sensi+std_sensi
	})

	data = [[s] for s in avg_list]
	table = wandb.Table(data=data, columns=["sensitivity"])
	wandb.log({'histogram': wandb.plot.histogram(table, "sensitivity",
							title="Histogram")})
		



