"""
Training script for DHFN
""" 
from run import DHFN_run

if __name__ == '__main__':
    DHFN_run(model_name='DHFN', dataset_name='mosi', is_tune=False, seeds=[1111], model_save_dir="./pt",
         res_save_dir="./result", log_dir="./log", mode='train', is_training=True)  