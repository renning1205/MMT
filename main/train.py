from train.train_fn.base import train_base, update_score_base
from train.train_fn.ride import train_ride, update_score_ride
from train.train_fn.ncl import train_ncl, update_score_ncl
from train.train_fn.bcl import train_bcl, update_score_bcl
from train.train_fn.base_kl import train_base_kl
from train.train_fn.kl_ride import train_kl_ride

def get_train_fn(args):
    if args.loss_fn == 'ride':
        return train_ride
    elif args.loss_fn == 'ncl':
        return train_ncl
    elif args.loss_fn == 'bcl':
        return train_bcl
    elif args.loss_fn == 'kl_ride':
        return train_kl_ride
    elif args.loss_fn.startswith('kl_'):
        return train_base_kl
    else:
        return train_base

        
        
def get_update_score_fn(args):
    if args.loss_fn == 'ride':
        return update_score_ride
    elif args.loss_fn == 'ncl':
        return update_score_ncl
    elif args.loss_fn == 'bcl':
        return update_score_bcl
    else:
        return update_score_base


