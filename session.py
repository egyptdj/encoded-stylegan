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
        train_summary_writer = tf.summary.FileWriter(result_dir+'/summary/train')
        val_summary_writer = tf.summary.FileWriter(result_dir+'/summary/validation')
        original_image_summary = sess.run(self.graph.test_original_image_summary)
        val_summary_writer.add_summary(original_image_summary)
        for iter in range(num_iter):
            _, scalar_summary = sess.run([self.graph.optimize, self.graph.scalar_summary], {self.graph.learning_rate: learning_rate})
            train_summary_writer.add_summary(scalar_summary, iter)
            if iter%save_iter==0: self.graph.saver.save(sess, result_dir+'/model/encoded_stylegan.ckpt') # save model
            if iter%10000==0 and iter!=0: learning_rate *= 0.98 # decay learning rate
            if iter%1000==0: # print/add summary
                train_image_summary, test_image_summary = sess.run(self.train_image_summary+self.test_recovered_image_summary)
                train_summary_writer.add_summary(train_image_summary, iter)
                val_summary_writer.add_summary(test_image_summary, iter)
        train_image_summary, test_image_summary = sess.run(self.train_image_summary+self.test_recovered_image_summary)
        train_summary_writer.add_summary(train_image_summary, iter)
        val_summary_writer.add_summary(test_image_summary, iter)
        self.graph.saver.save(sess, result_dir+'/model/encoded_stylegan.ckpt')
