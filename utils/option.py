import os
import csv
import argparse


def parse():
    parser = argparse.ArgumentParser(description='- Encoded StyleGAN -')
    parser.add_argument('-dT', '--testim_dir', type=str, default='images', help='directory path to load dataset')
    parser.add_argument('-dR', '--result_dir', type=str, default='results', help='directory path to save the trained model and/or the resulting image')
    parser.add_argument('-dC', '--cache_dir', type=str, default='cache', help='directory path to save/load cache files')
    parser.add_argument('-dM', '--model_dir', type=str, default='model', help='directory path to load models files')
    opt_dict = vars(parser.parse_args())

    return opt_dict
