import os
import csv
import argparse


def parse():
    parser = argparse.ArgumentParser(description='- Encoded StyleGAN -')
    parser.add_argument('-exp', '--exp_name', type=str, default='exp', help='experiment name')
    parser.add_argument('-b', '--minibatch_size', type=int, default=4, help='number of data to place in a minibatch')
    parser.add_argument('-l', '--learning_rate', type=float, default=1e-3, help='learning rate of the encoder training')
    parser.add_argument('-u', '--uniform_noise', type=bool, default=False, help='use random uniform noise for image generation')
    parser.add_argument('-nr', '--noise_range', type=float, default=1.0, help='noise range')
    parser.add_argument('-lE', '--encoding_lambda', type=float, default=0.0, help='encoding coefficient')
    parser.add_argument('-lV', '--vgg_lambda', type=float, default=0.0, help='vgg perceptual coefficient')
    parser.add_argument('-lLP', '--lpips_lambda', type=float, default=0.0, help='lpips coefficient')
    parser.add_argument('-lLAP', '--laploss_lambda', type=float, default=0.0, help='laploss coefficient')
    parser.add_argument('-l2', '--l2_lambda', type=float, default=0.0, help='l2 coefficient')
    parser.add_argument('-l1i', '--l1_image_lambda', type=float, default=0.0, help='l1 image coefficient')
    parser.add_argument('-l1l', '--l1_latent_lambda', type=float, default=0.0, help='l1 latent coefficient')
    parser.add_argument('-ms', '--modeseek', type=float, default=0.0, help='use modeseek regularizer coefficient')
    parser.add_argument('-nlin', '--nonlinearity', type=str, default='lrelu', help='nonlinearity [lrelu/relu]')
    parser.add_argument('-wsc', '--use_wscale', type=bool, default=True, help='use wscale')
    parser.add_argument('-mbstd', '--mbstd_group_size', type=int, default=4, help='mbstd group size')
    parser.add_argument('-mbfeat', '--mbstd_num_features', type=int, default=1, help='mbstd num features')
    parser.add_argument('-blur', '--blur_filter', type=bool, default=False, help='blur filter')
    parser.add_argument('-fs', '--fused_scale', type=str, default='auto', help='nonlinearity [lrelu/relu]')
    parser.add_argument('-dT', '--dataset_generated', type=bool, default=False, help='generated dataset or FFHQ')
    parser.add_argument('-dD', '--data_dir', type=str, default='/media/bispl/workdisk/FFHQ_flickrface/tfrecords', help='directory path to load dataset')
    parser.add_argument('-dV', '--validation_dir', type=str, default='images/validation', help='directory path to load dataset')
    parser.add_argument('-dR', '--result_dir', type=str, default='results', help='directory path to save the trained model and/or the resulting image')
    parser.add_argument('-dC', '--cache_dir', type=str, default='cache', help='directory path to save cache files')
    parser.add_argument('-iN', '--num_iter', type=int, default=10000000, help='total number of iterations to train')
    parser.add_argument('-iS', '--save_iter', type=int, default=1000, help='save model at every specified iterations')
    opt_dict = vars(parser.parse_args())

    opt_dict['result_dir'] = os.path.join(opt_dict['result_dir'], opt_dict['exp_name'])

    if not os.path.exists(opt_dict['result_dir']): os.makedirs(opt_dict['result_dir'])
    with open(os.path.join(opt_dict['result_dir'],"argv.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(opt_dict.items())

    return opt_dict
