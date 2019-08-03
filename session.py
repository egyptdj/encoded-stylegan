import tqdm
import tensorflow as tf
from stylegan.dnnlib import tflib


class SessionEncodedStyleGAN(object):
    def __init__(self):
        super(SessionEncodedStyleGAN, self).__init__()
        self.is_built = False

    def build(self, graph):
        assert graph.is_built
        self.train = [graph.optimize, graph.summary, graph.total_loss, graph.mse_loss, graph.perceptual_loss, graph.psnr, graph.ssim]
        self.test = [graph.summary, graph.total_loss, graph.mse_loss, graph.perceptual_loss, graph.psnr, graph.ssim]
        self.recover_image = [graph.recovered_image]
        self.input_image = [graph.input_image]
        self.saver = graph.saver
        self.is_built = True

    def train(self, learning_rate, num_iter, save_iter, result_dir):
        sess = tf.get_default_session()
        tflib.tfutil.init_uninitialized_vars()
        summary_writer = tf.summary.FileWriter(result_dir, graph=tf.get_default_graph())
        learning_rate_tensor = tf.get_variable('learning_rate')
        sess.run(learning_rate_tensor.assign(learning_rate))
        for iter in range(num_iter):
            _, summary, total_loss, mse_loss, perceptual_loss = sess.run(self.train)
            print('iter {:7d} | total {:.4f} | mse {:.4f} | percep {:.4f}'.format(iter, total_loss, mse_loss, perceptual_loss))
            if iter%save_iter==0:
                summary_writer.add_summary(summary)
                self.saver.save(sess, result_dir+"/encoded_stylegan.ckpt")
        summary_writer.add_summary(summary)
        self.saver.save(sess, result_dir+"/encoded_stylegan.ckpt")
