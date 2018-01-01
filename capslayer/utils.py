import os
import scipy
import numpy as np
import tensorflow as tf


def reduce_sum(input_tensor, axis=None, keepdims=False, name=None):
    try:
        return tf.reduce_sum(input_tensor, axis=axis, keepdims=keepdims, name=name)
    except:
        return tf.reduce_sum(input_tensor, axis=axis, keep_dims=keepdims, name=name)


def softmax(logits, axis=None, name=None):
    try:
        return tf.nn.softmax(logits, axis=axis, name=name)
    except:
        return tf.nn.softmax(logits, dim=axis, name=name)


def euclidean_norm(input, axis=2, keepdims=True, epsilon=True):
    if epsilon:
        norm = tf.sqrt(reduce_sum(tf.square(input), axis=axis, keepdims=keepdims) + 1e-9)
    else:
        norm = tf.sqrt(reduce_sum(tf.square(input), axis=axis, keepdims=keepdims))

    return(norm)


def load_mnist(batch_size, is_training=True):
    path = os.path.join('models', 'data', 'mnist')
    if is_training:
        fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainX = loaded[16:].reshape((60000, 784)).astype(np.float32)

        fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainY = loaded[8:].reshape((60000)).astype(np.int32)

        trX = trainX[:55000] / 255.
        trY = trainY[:55000]

        valX = trainX[55000:, ] / 255.
        valY = trainY[55000:]

        num_tr_batch = 55000 // batch_size
        num_val_batch = 5000 // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
    else:
        fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 784)).astype(np.float)

        fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.int32)

        num_te_batch = 10000 // batch_size
        return teX / 255., teY, num_te_batch


def load_fashion_mnist(batch_size, is_training=True):
    path = os.path.join('models', 'data', 'fashion-mnist')
    if is_training:
        fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainX = loaded[16:].reshape((60000, 784)).astype(np.float32)

        fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainY = loaded[8:].reshape((60000)).astype(np.int32)

        trX = trainX[:55000] / 255.
        trY = trainY[:55000]

        valX = trainX[55000:, ] / 255.
        valY = trainY[55000:]

        num_tr_batch = 55000 // batch_size
        num_val_batch = 5000 // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
    else:
        fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 784)).astype(np.float)

        fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.int32)

        num_te_batch = 10000 // batch_size
        return teX / 255., teY, num_te_batch

celeb_trainX = 0
celeb_trainY = 0
def load_celebgender(batch_size, is_training=True):
    import pickle
    from PIL import Image

    tf.logging.info("CALLED load_celebgender")
    path = os.path.join('models', 'data', 'celeb')
    ALL_IMGS = np.load((os.path.join(path, "celeb_images.npz")))
    IMAGE_SIZE = 100
    def load_celebimg(hsh):
        
        imgcontent = Image.open(ALL_IMGS[hsh].all())
        imgcontent = imgcontent.resize((IMAGE_SIZE,IMAGE_SIZE))
        img = np.fromstring(imgcontent.tobytes(), dtype=np.uint8)
        img = img.reshape((IMAGE_SIZE*IMAGE_SIZE*3))
        img = img.astype('float32')/255.0
        return img
    
    if is_training:
        fd = open(os.path.join(path, 'celeb_train.dict'),'rb')
        TRAIN_KEYS = pickle.load(fd)

        global celeb_trainX, celeb_trainY
        if type(celeb_trainX) == int:
            celeb_trainX = np.array([load_celebimg(x) for x in TRAIN_KEYS.keys()])
            celeb_trainY = np.array(list(TRAIN_KEYS.values())).astype(np.int32)

        # TOTAL of 3676
        trX = celeb_trainX[:3300]
        trY = celeb_trainY[:3300]

        valX = celeb_trainX[3300:, ]
        valY = celeb_trainY[3300:]

        num_tr_batch = len(trX) // batch_size
        num_val_batch = len(valX) // batch_size
        tf.logging.info("FINISHED CALL load_celebgender")
        return trX, trY, num_tr_batch, valX, valY, num_val_batch
    else:
        fd = open(os.path.join(path, 'celeb_test.dict'),'rb')
        TEST_KEYS = pickle.load(fd)

        testX = np.array([load_celebimg(x) for x in TEST_KEYS.keys()])
        testY = np.array(list(TEST_KEYS.values())).astype(np.int32)

        num_te_batch = len(testX) // batch_size
        return testX, testY, num_te_batch

def load_smallNORB(batch_size, is_training=True):
    pass


def load_data(dataset, batch_size, is_training=True, one_hot=False):
    if dataset == 'mnist':
        return load_mnist(batch_size, is_training)
    elif dataset == 'fashion-mnist':
        return load_fashion_mnist(batch_size, is_training)
    elif dataset == 'celebgender':
        return load_celebgender(batch_size, is_training)
    elif dataset == 'smallNORB':
        return load_smallNORB(batch_size, is_training)
    else:
        raise Exception('Invalid dataset, please check the name of dataset:', dataset)


def get_batch_data(dataset, batch_size, num_threads):
    if dataset == 'mnist':
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_mnist(batch_size, is_training=True)
    elif dataset == 'fashion-mnist':
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_fashion_mnist(batch_size, is_training=True)
    elif dataset == 'celebgender':
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_celebgender(batch_size, is_training=True)
    elif dataset == 'smallNORB':
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_smallNORB(batch_size, is_training=True)
    
    print("Size of TrX: "+str(trX.nbytes)+" bytes")
    data_queues = tf.train.slice_input_producer([trX, trY])
    X, Y = tf.train.shuffle_batch(data_queues, num_threads=num_threads,
                                  batch_size=batch_size,
                                  capacity=batch_size * 4,
                                  min_after_dequeue=batch_size * 2,
                                  allow_smaller_final_batch=False)

    return(X, Y)


def save_images(imgs, size, path):
    '''
    Args:
        imgs: [batch_size, image_height, image_width]
        size: a list with tow int elements, [image_height, image_width]
        path: the path to save images
    '''
    imgs = (imgs + 1.) / 2  # inverse_transform
    return(scipy.misc.imsave(path, mergeImgs(imgs, size)))


def mergeImgs(images, size):
    h, w = images.shape[1], images.shape[2]
    imgs = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        imgs[j * h:j * h + h, i * w:i * w + w, :] = image

    return imgs


def get_transformation_matrix_shape(in_pose_shape, out_pose_shape):
    return([out_pose_shape[0], in_pose_shape[0]])
