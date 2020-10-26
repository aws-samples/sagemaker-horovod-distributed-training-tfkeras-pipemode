# NOTE: should run in conda tensorflow_p36 (TF1) env.
import tensorflow as tf
import os
import sys
import shutil
import argparse
                
def split_tfrecord(data_dir, tfrecord_path, split_size, num_of_instances):
    with tf.Graph().as_default(), tf.Session() as sess:
        ds = tf.data.TFRecordDataset(tfrecord_path).batch(split_size)
        batch = ds.make_one_shot_iterator().get_next()

        part_num = 0
                    
        while True:
            try:
                records = sess.run(batch)
                
                shard_bucket_index = int(part_num/num_of_instances)
                
                shard_dir = os.path.join(data_dir, f'train/{shard_bucket_index}')
                if not os.path.exists(shard_dir):
                    os.makedirs(shard_dir)

                output_file = os.path.join(shard_dir, f'train_{part_num}.tfrecords')
                print('Generating %s' % output_file)
                
                with tf.io.TFRecordWriter(output_file) as writer:
                    for record in records:
                        writer.write(record)
                part_num += 1
            except tf.errors.OutOfRangeError: break


def do_shard(data_dir, gpus_per_host, num_of_instances):
    tf.compat.v1.enable_eager_execution()
    
    input_file = os.path.join(data_dir, "train/train.tfrecords")
    raw_dataset = tf.data.TFRecordDataset(input_file)
    num_total_records = sum(1 for _ in raw_dataset)
    
    if num_total_records % (num_of_instances * gpus_per_host) != 0:
        print('Error: Number of total tfrecords ({}) are not a multiple of number of shards ({})!' % (num_total_records, gpus_per_host))
        sys.exit(-1)
    else:
        size = num_total_records // ( num_of_instances * gpus_per_host)
        split_tfrecord(data_dir, input_file, size, num_of_instances)
        
if __name__ == '__main__':        
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-dir',
        type=str,
        default='',
        help='Directory to download and extract CIFAR-10 to.')
    parser.add_argument(
        '--gpus-per-host',
        type=int,
        default=1,
        help='Number of GPUs of an instance for Horovod.')
    parser.add_argument(
        '--num-of0instances',
        type=int,
        default=1,
        help='Number of instances for Horovod.')

    args = parser.parse_args()
    do_shard(args.data_dir, args.gpus_per_host, args.num_of_instances)
