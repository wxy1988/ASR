# coding=utf-8
import os
import time
from argparse import ArgumentParser
import yaml

from evaluate import Evaluator
from model_tfrecord import *
from utils import DataReader, AttrDict, available_variables, expand_feed_dict

def parse_function_var(example_proto):
    keys_to_features = {'feat_shape': tf.FixedLenFeature(shape=(1,2), dtype=tf.int64),
                        'feat': tf.VarLenFeature(tf.float32),
                        'label_shape': tf.FixedLenFeature(shape=(1,), dtype=tf.int64),
                        'label': tf.VarLenFeature(tf.int64)}
    
    parsed_feature = tf.parse_single_example(example_proto, keys_to_features)
    return parsed_feature['feat_shape'], parsed_feature['feat'], parsed_feature['label_shape'], parsed_feature['label']

def train(config):
    logger = logging.getLogger('')

    """Train a model with a config file."""
    train_graph = tf.Graph()
    print(config.train.tfrecord_pattern)
    data_files = tf.gfile.Glob(config.train.tfrecord_pattern)
    logging.info("Find {} tfrecords files".format(len(data_files)))
    with train_graph.as_default():
        data_holder = tf.placeholder(tf.string, shape = [None])
        dataset = tf.data.TFRecordDataset(data_holder, num_parallel_reads = config.train.read_threads)
        dataset = dataset.map(parse_function_var, num_parallel_calls = config.train.read_threads)
        shuffle_data = True
        if shuffle_data is True:
            dataset = dataset.shuffle(buffer_size = 10000)
        dataset = dataset.repeat(config.train.num_epochs).batch(config.train.batchsize_read)

        iterator = dataset.make_initializable_iterator()

        feat_shape_tensor, feat_tensor, label_shape_tensor, label_tensor = iterator.get_next()

        feat_tensor = tf.sparse_tensor_to_dense(feat_tensor)
        label_tensor = tf.sparse_tensor_to_dense(label_tensor)
        label_tensor = tf.cast(label_tensor, tf.int32)
        feat_tensor_shapeop = tf.shape(feat_tensor)
        feat_tensor = tf.reshape(feat_tensor, [feat_tensor_shapeop[0], -1, config.train.input_dim])
        
        

    model = eval(config.model)(config=config, num_gpus=config.train.num_gpus, X=feat_tensor, Y=label_tensor, tensor_graph=train_graph)
    model.build_train_model(test=config.train.eval_on_dev)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement = True

    summary_writer = tf.summary.FileWriter(config.model_dir, graph=model.graph)


    with tf.Session(config=sess_config, graph=model.graph) as sess:
        # Initialize all variables.
        sess.run(tf.global_variables_initializer())

        sess.run(iterator.initializer, feed_dict = {data_holder : data_files})
        # Reload variables in disk.
        if tf.train.latest_checkpoint(config.model_dir):
            available_vars = available_variables(config.model_dir)
            if available_vars:
                saver = tf.train.Saver(var_list=available_vars)
                saver.restore(sess, tf.train.latest_checkpoint(config.model_dir))
                for v in available_vars:
                    logger.info('Reload {} from disk.'.format(v.name))
            else:
                logger.info('Nothing to be reload from disk.')
        else:
            logger.info('Nothing to be reload from disk.')


        global dev_bleu, toleration
        dev_bleu = 0
        toleration = config.train.toleration

        def train_one_step(batch):
            feat_batch, target_batch,batch_size = batch
            feed_dict = expand_feed_dict({model.src_pls: feat_batch,
                                          model.dst_pls: target_batch})
            step, lr, loss, _ = sess.run(
                [model.global_step, model.learning_rate,
                 model.loss, model.train_op],
                feed_dict=feed_dict)
            if step % config.train.summary_freq == 0:
                summary = sess.run(model.summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary, global_step=step)
            return step, lr, loss

        def maybe_save_model():
            global dev_bleu, toleration
            new_dev_bleu = dev_bleu + 1
            if new_dev_bleu >= dev_bleu:
                mp = config.model_dir + '/model_step_{}'.format(step)
                model.saver.save(sess, mp)
                logger.info('Save model in %s.' % mp)
                toleration = config.train.toleration
                dev_bleu = new_dev_bleu
            else:
                toleration -= 1

        step = 0
        while True:
            try:
                pre_train_time = time.time()
                feat_shape, feat, label_shape, label = sess.run([feat_shape_tensor, feat_tensor, label_shape_tensor, label_tensor])
                batch = (feat, label, feat.shape[0])
                #logging.info("This batch has {} samples".format(feat.shape[0]))
                #logging.info("The feat shape is {}".format(feat.shape))
                # Train normal instances.
                start_time = time.time()
                step, lr, loss = train_one_step(batch)
                logger.info(
                            'step: {0}\tlr: {1:.6f}\tloss: {2:.4f}\ttrain_time: {3:.4f}\tpre_train_time: {4:.5f}\tbatch_size: {5}'.
                            format(step, lr, loss, time.time() - start_time, start_time - pre_train_time, batch[2]))
                # Save model
                pre_train_time = time.time()
                if config.train.save_freq > 0 and step % config.train.save_freq == 0:
                    maybe_save_model()
                
                if config.train.num_steps and step >= config.train.num_steps:
                    break

                # Save model per epoch if config.train.save_freq is less or equal than zero
                if config.train.save_freq <= 0:
                    maybe_save_model()

                # Early stop
                if toleration <= 0:
                    break
            except tf.errors.OutOfRangeError:
                logging.info("All data done!")
        logger.info("Finish training.")

def config_logging(log_file):
    import logging
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S', filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


if __name__ == '__main__':
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    parser = ArgumentParser()
    parser.add_argument('-c', '--config', dest='config')
    args = parser.parse_args()
    #args.config = './config_template_char_unit512_block6_left3_big_dim80_sp.yaml'
    print(args.config)
    config = AttrDict(yaml.load(open(args.config)))
    # Logger
    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)
    config_logging(config.model_dir + '/train.log')
    import shutil
    shutil.copy(args.config, config.model_dir)
    import datetime
    time_stamp = datetime.datetime.now()
    print("start_time:" + time_stamp.strftime('%Y.%m.%d-%H:%M:%S'))
    try:
        # Train
        train(config)
    except Exception as e:
        import traceback
        logging.error(traceback.format_exc())
    time_stamp = datetime.datetime.now()
    print("end_time:" + time_stamp.strftime('%Y.%m.%d-%H:%M:%S'))
