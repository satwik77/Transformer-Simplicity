import torch 
from torch import nn
import numpy as np
import ipdb as pdb
import random
import pandas as pd

from attrdict import AttrDict
import argparse
import wandb

from src.components.sanplay import TransformerCLF

from src.other_utils import sample_bstr, flip_i, Vocab, sents_to_idx


out_file= 'out/sensi_gauss_exp.csv'
out_grouped= 'out/sensi_gauss_group.csv'

parser = argparse.ArgumentParser(description='')
parser.add_argument('-gpu', type=int, default=7, help='Specify the gpu to use')
parser.add_argument('-run', type=str, default='trialGaussA', help='run name')
parser.add_argument('-sample_size', type=int, default=50, help='Sample size for estimation of each random initialization')  # Use 1000 or above 

parser.add_argument('-len', type=int, default=15, help='Length of inputs')
parser.add_argument('-std_init', type=float, default=10.0, help='weight init st deviation ')
parser.add_argument('-i_init', type=float, default=1.0, help='Range of input values [-B,+B]')
parser.add_argument('-d_model', type=int, default=64, help='SAN d_model')
parser.add_argument('-depth', type=int, default=2, help='Layers')
parser.add_argument('-head', type=int, default=8, help='heads')
parser.add_argument('-pos_encode', type=str, default='learnable', choices= ['absolute', 'learnable'],  help='Postional encoding type')

parser.add_argument('-trials', type=int, default=4, help='Number of trials or randomly initialized models')


parser.add_argument('-bias', dest='bias', action='store_true', help='Keep bias')
parser.add_argument('-no-bias', dest='bias', action='store_false', help='Do not keep bias')
parser.set_defaults(bias=False)

parser.add_argument('-wandb', dest='wandb', action='store_true', help='Store wandb')
parser.add_argument('-no-wandb', dest='wandb', action='store_false', help='Do not store wandb')
parser.set_defaults(wandb=False)




args = parser.parse_args()


run_name = args.run + '_'+ 'h{}d{}t{}p{}'.format(args.d_model, args.depth, args.trials, args.pos_encode) + '_'+ str(np.random.choice(10000))

if args.gpu >=0:
    device =torch.device('cuda:{}'.format(args.gpu))
else:
    device = torch.device('cpu')



########## Wandb Init ###############

if args.wandb:
    metrics = dict(
        depth= args.depth,
        d_model = args.d_model,
        length= args.len,
        sample_size= args.sample_size,
        trials = args.trials,
        weight_range= args.std_init,
        pos_encode= args.pos_encode,
    )

    wandb.init(
        project= 'RandGaussSensi-SAN-B',
        name= run_name,
        config= metrics,
        group= args.pos_encode,
    )

########################################

def test_model(str_list, clf):
    clf.eval()
    str_list = [x + 's' for x in str_list]
    wlens = torch.tensor([len(x) for x in str_list])
    batch_ids = sents_to_idx(voc, str_list)
    str_ids = batch_ids[:,:-1].transpose(0,1)
    str_ids = str_ids.to(device)
    wlens= wlens.to(device)
    
    with torch.no_grad():
        output = clf(str_ids, wlens)
        preds = output.cpu().numpy()
        preds= preds.reshape(-1)
        preds = np.array(preds>=0.5, dtype=int)
        # preds = preds.argmax(axis=1)
    return preds
    



### Model Def #####
print('Loading Model')
inp_len =args.len
pos_encode= args.pos_encode
d_model = args.d_model
d_ffn= 2*d_model
depth= args.depth
std_init= args.std_init
inp_init = args.i_init


clf = TransformerCLF(ntoken =4, noutputs=1,  d_model=d_model, nhead=args.head, d_ffn=d_ffn, nlayers=depth, dropout=0.0, pos_encode_type =pos_encode)
# clf = RNNModel(ntoken =3, noutputs=1, nemb= 2, nhid=args.hidden_size, nlayers= args.depth, rnn_type=rnn_type, nonlinearity=nonlinearity)
clf = clf.to(device)
voc = Vocab()


### Sensitivity Experiment ####

sample_size = args.sample_size
check_idx = list(range(inp_len))

### Pandas Data setup

cols = ['run_name', 'sample_size', 'len', 'd_model', 'depth', 'heads', 'pos_encode', 'is_bias', 'std_init', 'inp_init', 'trial', 'sensi_count', 'avg_sensi']
df= pd.DataFrame(data=[], columns=cols)
fixed_data= [run_name, sample_size, inp_len, d_model, depth, args.head, pos_encode, False, std_init, inp_init]

cols_group = ['run_name', 'sample_size', 'num_trials', 'len', 'd_model', 'depth', 'heads', 'pos_encode', 'is_bias', 'std_init', 'inp_init', 'avg_sensi', 'std_sensi']
big_df = pd.DataFrame(data=[], columns=cols_group)
group_data= [run_name, sample_size, args.trials, inp_len, d_model, depth, args.head, pos_encode, False, std_init, inp_init]


print('Generating Samples')
sample_points  =sample_bstr(sample_size, length= inp_len)

avg_list= []

for trial in range(args.trials):
    print('Running for trial : {}'.format(trial))
    clf.init_gauss_weights(std_init= std_init, inp_init=inp_init, dec_init= 1.0)

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

    print('\n-------------Trial {} Done--------------'.format(trial))
    print(sensi_count)
    print(avg_sensi)

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
        'mean+std':mean_sensi+std_sensi
    })

    data = [[s] for s in avg_list]
    table = wandb.Table(data=data, columns=["sensitivity"])
    wandb.log({'histogram': wandb.plot.histogram(table, "sensitivity",
                            title="Histogram")})
        



