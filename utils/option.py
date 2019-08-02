import os
import csv
import argparse


def parse():
    parser = argparse.ArgumentParser(description='- Encoded StyleGAN -')
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='number of data to place in a minibatch')
    parser.add_argument('-nE', '--exp_name', type=str, default='exp', help='experiment name')
    parser.add_argument('-lG', '--learning_rate', type=float, default=1e-4, help='learning rate of the encoder training')
    parser.add_argument('-dD', '--data_dir', type=str, default='/media/bispl/workdisk/FFHQ_flickrface/tfrecords', help='directory path to load dataset')
    parser.add_argument('-dR', '--result_dir', type=str, default='results', help='directory path to save the trained model and/or the resulting image')
    parser.add_argument('-dC', '--cache_dir', type=str, default='cache', help='directory path to save cache files')
    parser.add_argument('-eN', '--num_epoch', type=int, default=1000, help='total number of epochs to train')
    parser.add_argument('-eS', '--save_epoch', type=int, default=50, help='save model at every specified epochs')
    opt_dict = vars(parser.parse_args())

    opt_dict['result_dir'] = os.path.join(opt_dict['result_dir'], opt_dict['exp_name'])

    if not os.path.exists(opt_dict['result_dir']): os.makedirs(opt_dict['result_dir'])
    with open(os.path.join(opt_dict['result_dir'],"argv.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(opt_dict.items())

    return opt_dict
