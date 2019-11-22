import os
import csv
import argparse


def parse():
    parser = argparse.ArgumentParser(description='- Encoded StyleGAN -')
    parser.add_argument('-nE', '--exp_name', type=str, default='exp', help='experiment name')
    parser.add_argument('-g', '--num_gpus', type=int, default=1, help='number of gpus to use')
    parser.add_argument('-b', '--minibatch_size', type=int, default=2, help='number of data to place in a minibatch')
    parser.add_argument('-lE', '--encoder_learning_rate', type=float, default=1e-3, help='learning rate of the encoder training')
    parser.add_argument('-lG', '--generator_learning_rate', type=float, default=1e-3, help='learning rate of the encoder training')
    parser.add_argument('-u', '--uniform_noise', type=bool, default=False, help='use random uniform noise for image generation')
    parser.add_argument('-nr', '--noise_range', type=float, default=1.0, help='noise range')
    parser.add_argument('-sV', '--vgg_shape', type=int, default=224, help='vgg input reshape size')
    parser.add_argument('-lV', '--vgg_lambda', type=float, default=1.0, help='vgg perceptual coefficient')
    parser.add_argument('-l2', '--l2_lambda', type=float, default=1.0, help='l2 coefficient')
    parser.add_argument('-l1', '--l1_lambda', type=float, default=1.0, help='l1 coefficient')
    parser.add_argument('--lpips_lambda', type=float, default=0.0, help='lpips coefficient')
    parser.add_argument('--mssim_lambda', type=float, default=0.0, help='mssim coefficient')
    parser.add_argument('--logcosh_lambda', type=float, default=0.0, help='logcosh coefficient')
    parser.add_argument('-st', '--structure', type=str, default='recursive', help='structure choice')
    parser.add_argument('-dD', '--data_dir', type=str, default='/media/bispl/workdisk/FFHQ_flickrface/tfrecords', help='directory path to load dataset')
    parser.add_argument('-dV', '--validation_dir', type=str, default='images/validation', help='directory path to load dataset')
    parser.add_argument('-dR', '--result_dir', type=str, default='results', help='directory path to save the trained model and/or the resulting image')
    parser.add_argument('-dC', '--cache_dir', type=str, default='cache', help='directory path to save cache files')
    parser.add_argument('-iN', '--num_iter', type=int, default=10000000, help='total number of iterations to train')
    parser.add_argument('-gp', '--gp_lambda', type=float, default=10.0, help='gradient penalty weight')
    parser.add_argument('-iLC', '--latent_critic_iter', type=int, default=5, help='number of critic iterations')
    parser.add_argument('-iIC', '--image_critic_iter', type=int, default=5, help='number of critic iterations')
    parser.add_argument('-iE', '--encoder_iter', type=int, default=1, help='number of encoder iterations')
    parser.add_argument('-iG', '--generator_iter', type=int, default=1, help='number of generator iterations')
    parser.add_argument('-iS', '--save_iter', type=int, default=1000, help='save model at every specified iterations')
    parser.add_argument('-iO', '--image_output', type=int, default=4, help='number of image summary output')
    parser.add_argument('--latent_critic_layers', type=int, default=8, help='number of z_critic layers')
    parser.add_argument('--progan', action='store_true', help='use progan model')
    parser.add_argument('--seed', type=int, default=0, help='random state seed')
    parser.add_argument('--dataset_generated', action='store_true', help='generated dataset or FFHQ')
    args = parser.parse_args()

    args.result_dir = os.path.join(args.result_dir, args.exp_name)

    os.makedirs(args.result_dir, exist_ok=True)
    with open(os.path.join(args.result_dir,"argv.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(vars(args).items())

    return args
