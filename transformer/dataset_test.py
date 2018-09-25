# coding=utf-8

import tensorflow as tf
import logging
import sys
import argparse

def test_dataset(data_path):
  data_files = tf.gfile.Glob(data_path)
  print(data_files)
  if len(data_files) == 0:
    logging.info("Error: Read no data files in path {}".format(data_path))
    sys.exit(1)

  dataset = tf.data.TFRecordDataset(data_files, num_parallel_reads=4)
  dataset = dataset.repeat(1).shuffle(10000).batch(32)  # just for testing

  iterator = dataset.make_one_shot_iterator()

  total_number = 0
  batch_feat, batch_label, batch_shape = iterator.get_next()
  print(batch_shape)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--record_path', type=str,
                      help='the path of tf record files')

  flags, _ = parser.parse_known_args()
  test_dataset(flags.record_path)

