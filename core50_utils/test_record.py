import tensorflow as tf

""""

tfrecord files are hardly debuggable, this is why I decided to create this script.
It's mainly designed for Core50 dataset, but it can easily be edited for any other tfrecord.

Example:

    python test_record.py --input="test.record"

"""""

tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
flags.DEFINE_string('input', '', 'tfrecord file path.')
FLAGS = flags.FLAGS

def explore_tf_record(path):
    for example in tf.python_io.tf_record_iterator(path):
        result = tf.train.Example.FromString(example)
        print("img_name: " +str(result.features.feature['image/filename'].bytes_list.value))
        print("xmin "+str(result.features.feature['image/object/bbox/xmin'].float_list.value))
        print("xmax "+str(result.features.feature['image/object/bbox/xmax'].float_list.value))
        print("ymin " +str(result.features.feature['image/object/bbox/ymin'].float_list.value))
        print("ymax " +str(result.features.feature['image/object/bbox/ymax'].float_list.value))
        print("CLASS "+str(result.features.feature['image/object/class/label'].int64_list.value))


if __name__ == '__main__':
    assert FLAGS.input, 'input is missing.'
    explore_tf_record(FLAGS.input)