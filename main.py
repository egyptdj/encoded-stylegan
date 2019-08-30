import os
import sys
import utils
import pickle
import numpy as np
import tensorflow as tf
import PIL.Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'stylegan'))
from stylegan import dnnlib
from stylegan.dnnlib import tflib
from stylegan.training.networks_stylegan import *


def restore_model(modeldir, cachedir, session):
    # RESTORE MODEL
    try:
        latest_checkpoint = tf.train.latest_checkpoint(os.path.join(modeldir, "model"))
        meta_graph = tf.train.import_meta_graph(".".join([latest_checkpoint, "meta"]))
        meta_graph.restore(sess=session, save_path=latest_checkpoint)
    except Exception as e:
        print (e)
        print("meta graph not found")
        return None

    url = os.path.join(cachedir, 'karras2019stylegan-ffhq-1024x1024.pkl')
    with open(url, 'rb') as f: _, _, Gs = pickle.load(f)


def load_image(imagedir):
    image_list = [image for image in os.listdir(imagedir) if image.endswith("png") or image.endswith("jpg") or image.endswith("jpeg")]
    assert len(image_list)>0

    imbatch = np.stack([np.array(PIL.Image.open(imagedir+"/"+image_path).resize((1024,1024))) for image_path in image_list], axis=0)/255.0
    return imbatch


def main():
    base_option = utils.option.parse()
    tflib.init_tf()
    sess = tf.get_default_session()
    restore_model(base_option['model_dir'], base_option['cache_dir'], sess)
    image_batch = load_image(base_option['testim_dir'])
    image_input, encoded_latent, image_output = tf.get_collection("TEST_NODES")
    result_image = sess.run(image_output, feed_dict={image_input: image_batch})
    # tflib.tfutil.init_uninitialized_vars()

    _ = tf.summary.image('target', tf.clip_by_value(tf.constant(image_batch), 0.0, 1.0), max_outputs=image_batch.shape[0], family='images', collections=['IMAGE_SUMMARY', tf.GraphKeys.SUMMARIES])
    _ = tf.summary.image('encoded', tf.clip_by_value(tf.transpose(tf.constant(result_image), perm=[0,2,3,1]), 0.0, 1.0), max_outputs=image_batch.shape[0], family='images', collections=['IMAGE_SUMMARY', tf.GraphKeys.SUMMARIES])
    summary = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(base_option['model_dir']+'/test_summary')
    # tflib.tfutil.init_uninitialized_vars()
    test_summary = sess.run(summary)
    summary_writer.add_summary(test_summary)

if __name__=='__main__':
    main()
