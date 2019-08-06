import tqdm
import tensorflow as tf
from stylegan.dnnlib import tflib


class SessionEncodedStyleGAN(object):
    def __init__(self):
        super(SessionEncodedStyleGAN, self).__init__()
        self.is_built = False

    def build(self, graph):
        assert graph.is_built
        self.graph = graph
        self.is_built = True

    def train(self, learning_rate, num_iter, save_iter, result_dir):
        sess = tf.get_default_session()
        tflib.tfutil.init_uninitialized_vars()
        searchlight_summary_writer = tf.summary.FileWriter(result_dir+'/summary/searchlight')
        encoder_summary_writer = tf.summary.FileWriter(result_dir+'/summary/encoder')

        for iter in range(num_iter):
            sess.run(self.graph.model.reset_searchlight)
            input_image = sess.run(self.graph.model.dataset.image)
            searchlight_lr = 1e-1
            for searchlight_iter in range(50):
                searchlight_lr * 0.9
                _, searchlight_scalar_summary = sess.run([self.graph.searchlight_optimize, self.graph.searchlight_scalar_summary], {self.graph.searchlight_learning_rate: searchlight_lr, self.graph.model.input: input_image})
                step=iter*10+searchlight_iter
                searchlight_summary_writer.add_summary(searchlight_scalar_summary, step)

            _, encoder_scalar_summary = sess.run([self.graph.encoder_optimize, self.graph.encoder_scalar_summary], {self.graph.encoder_learning_rate: learning_rate, self.graph.model.input: input_image})
            encoder_summary_writer.add_summary(encoder_scalar_summary, iter)
            if iter%save_iter==0:
                image_summary, total_loss, mse_loss, perceptual_loss, psnr, ssim = sess.run([self.graph.image_summary, self.graph.encoder_total_loss, self.graph.encoder_mse_loss, self.graph.encoder_perceptual_loss, self.graph.encoder_psnr, self.graph.encoder_ssim], {self.graph.model.input: input_image})
                encoder_summary_writer.add_summary(image_summary, iter)
                self.graph.saver.save(sess, result_dir+'/model/encoded_stylegan.ckpt')
                print('iter {:7d} | total {:.4f} | mse {:.4f} | percep {:.4f} | psnr {:.4f} ssim {:.4f}'.format(iter, total_loss, mse_loss, perceptual_loss, psnr, ssim))
        self.graph.saver.save(sess, result_dir+'/model/encoded_stylegan.ckpt')
