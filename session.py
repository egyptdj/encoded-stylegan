import tensorflow as tf
from tqdm import tqdm
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
        train_summary_writer = tf.summary.FileWriter(result_dir+'/summary/train')
        # val_summary_writer = tf.summary.FileWriter(result_dir+'/summary/validation')
        for iter in tqdm(range(num_iter)):
            if iter%100==0 or iter==0:
                original_image, encoded_latent = sess.run([self.graph.model.generator_output, self.graph.model.encoded_latent])
                _, iter_summary = sess.run([self.graph.optimize, self.graph.summary], {self.graph.learning_rate: learning_rate})
            else: _, iter_summary = sess.run([self.graph.optimize, self.graph.scalar_summary], {self.graph.learning_rate: learning_rate})
            if iter%save_iter==0: self.graph.saver.save(sess, result_dir+'/model/encoded_stylegan.ckpt') # save model
            if iter%10000==0 and iter!=0: learning_rate *= 0.98 # decay learning rate
            train_summary_writer.add_summary(iter_summary, iter)
        self.graph.saver.save(sess, result_dir+'/model/encoded_stylegan.ckpt')
