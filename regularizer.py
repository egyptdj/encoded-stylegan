import tensorflow as tf

def modeseek(image, latent):
    assert image.shape[0]==latent.shape[0]
    assert image.shape[0]>1
    _img1, _img2 = tf.split(image, 2, axis=0, name='image_split')
    _latent1, _latent2 = tf.split(latent, 2, axis=0, name='latent_split')
    modeseek_loss = tf.reduce_mean(tf.abs(_img1-_img2), name='generated_image_distance') / tf.reduce_mean(tf.abs(_latent1-_latent2), name='latent_distance')
    min_modeseek_loss = 1 / (modeseek_loss + 1e-8)
    return min_modeseek_loss
