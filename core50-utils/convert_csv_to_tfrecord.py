import tensorflow as tf
import pandas

core50_path = 'core50.csv'
tf_record_output = 'train.record'

csv = pandas.read_csv(core50_path).values
with tf.python_io.TFRecordWriter(tf_record_output) as writer:
    for row in csv:
        img_path,width, height, xmin, ymin, xmax, ymax , label = row[0],row[1], row[2], row[3], row[4], row[5], row[6], row[7]

        # Strings must be bytes, this why encode is needed.
        img_path = img_path.encode('utf8')

        example = tf.train.Example()
        example.features.feature["Filename"].bytes_list.value.append(img_path)
        example.features.feature["width"].int64_list.value.append(width)
        example.features.feature["height"].int64_list.value.append(height)
        example.features.feature["xmin"].float_list.value.append(xmin)
        example.features.feature["ymin"].float_list.value.append(ymin)
        example.features.feature["xmax"].float_list.value.append(xmax)
        example.features.feature["ymax"].float_list.value.append(ymax)
        example.features.feature["label"].int64_list.value.append(label)
        writer.write(example.SerializeToString())


print('Done! Your .record file is ready.')
