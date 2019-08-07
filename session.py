import tqdm
import tensorflow as tf
from stylegan.dnnlib import tflib


class SessionEncodedStyleGAN(object):
    def __init__(self):
        super(SessionEncodedStyleGAN, self).__init__()
        self.is_built = False

    def build(self, graph):
        assert graph.is_built
        # self.graph = graph
        self.train_op = [graph.optimize, graph.scalar_summary, graph.total_loss, graph.mse_loss, graph.perceptual_loss, graph.psnr, graph.ssim]
        self.test_op = [graph.summary, graph.total_loss, graph.mse_loss, graph.perceptual_loss, graph.psnr, graph.ssim]
        self.recover_image_op = [graph.recovered_image]
        self.original_image_op = [graph.original_image]
        self.image_summary = graph.image_summary
        self.learning_rate = graph.learning_rate
        self.saver = graph.saver
        self.is_built = True

    def train(self, learning_rate, num_iter, save_iter, result_dir):
        sess = tf.get_default_session()
        tflib.tfutil.init_uninitialized_vars()
        summary_writer = tf.summary.FileWriter(result_dir+'/summary')
        for iter in range(num_iter):
            if iter%10000==0 and iter!=0: learning_rate *= 0.5
            _, scalar_summary, total_loss, mse_loss, perceptual_loss, psnr, ssim = sess.run(self.train_op, {self.learning_rate: learning_rate})
            summary_writer.add_summary(scalar_summary, iter)
            if iter%save_iter==0:
                print('iter {:7d} | total {:.4f} | mse {:.4f} | percep {:.4f} | psnr {:.4f} ssim {:.4f}'.format(iter, total_loss, mse_loss, perceptual_loss, psnr, ssim))
                image_summary = sess.run(self.image_summary)
                summary_writer.add_summary(image_summary, iter)
                self.saver.save(sess, result_dir+'/model/encoded_stylegan.ckpt')
        image_summary = sess.run(self.image_summary)
        summary_writer.add_summary(image_summary, iter)
        self.saver.save(sess, result_dir+'/model/encoded_stylegan.ckpt')
