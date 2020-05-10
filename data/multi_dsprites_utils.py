# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Multi-dSprites dataset reader."""

import functools
import tensorflow as tf
COMPRESSION_TYPE = tf.io.TFRecordOptions.get_compression_type_string('GZIP')
IMAGE_SIZE = [64, 64]
# The maximum number of foreground and background entities in each variant
# of the provided datasets. The values correspond to the number of
# segmentation masks returned per scene.
MAX_NUM_ENTITIES = {
    'binarized': 4,
    'colored_on_grayscale': 6,
    'colored_on_colored': 5
}
BYTE_FEATURES = ['mask', 'image']
def feature_descriptions(max_num_entities, is_grayscale=False):
    """Create a dictionary describing the dataset features.
    Args:
      max_num_entities: int. The maximum number of foreground and background
        entities in each image. This corresponds to the number of segmentation
        masks and generative factors returned per scene.
      is_grayscale: bool. Whether images are grayscale. Otherwise they're assumed
        to be RGB.
    Returns:
      A dictionary which maps feature names to `tf.Example`-compatible shape and
      data type descriptors.
    """

    num_channels = 1 if is_grayscale else 3
    return {
        'image': tf.io.FixedLenFeature(IMAGE_SIZE+[num_channels], tf.string), #shape dtype
        'mask': tf.io.FixedLenFeature(IMAGE_SIZE+[max_num_entities, 1], tf.string),
        'x': tf.io.FixedLenFeature([max_num_entities], tf.float32),
        'y': tf.io.FixedLenFeature([max_num_entities], tf.float32),
        'shape': tf.io.FixedLenFeature([max_num_entities], tf.float32),
        'color': tf.io.FixedLenFeature([max_num_entities, num_channels], tf.float32),
        'visibility': tf.io.FixedLenFeature([max_num_entities], tf.float32),
        'orientation': tf.io.FixedLenFeature([max_num_entities], tf.float32),
        'scale': tf.io.FixedLenFeature([max_num_entities], tf.float32),
    }

def _decode(example_proto, features):
    # Parse the input `tf.Example` proto using a feature description dictionary.
    single_example = tf.io.parse_single_example(example_proto, features)
    for k in BYTE_FEATURES: #mask image
        single_example[k] = tf.squeeze(tf.decode_raw(single_example[k], tf.uint8),
                                     axis=-1)  # height width entities channels
    # To return masks in the canonical [entities, height, width, channels] format,
    # we need to transpose the tensor axes.
    single_example['mask'] = tf.transpose(single_example['mask'], [0, 1, 3, 2]) #H W 1 M

    return map(single_example)

def map(x):
    img = x['image']
    img = tf.cast(img, tf.float32)
    img = img/255
    mask = x['mask']
    mask = tf.cast(mask, tf.float32)
    mask = mask/255 #0~1
    data = {}
    data['img'] = img
    data['masks'] = mask
    return data

def dataset(tfrecords_path, batch_size, phase='train'):
    if phase=='test':
        skipnum, takenum = 0,2000
        shuffle = False
    elif phase=='val':
        skipnum, takenum = 2000,1000
        shuffle = False 
    else:
        skipnum, takenum = 3000, -1
        shuffle = True
    max_num_entities = MAX_NUM_ENTITIES['colored_on_colored'] #colored on colored -> 5
    raw_dataset = tf.data.TFRecordDataset(
        tfrecords_path, compression_type=COMPRESSION_TYPE)
    raw_dataset = raw_dataset.skip(skipnum).take(takenum)
    features = feature_descriptions(max_num_entities, False)
    partial_decode_fn = functools.partial(_decode, features=features)
    
    dataset = raw_dataset.map(partial_decode_fn,num_parallel_calls=1)
    if shuffle:
        dataset = dataset.shuffle(seed=479, buffer_size=50000, reshuffle_each_iteration=True)
    dataset = dataset.repeat().batch(batch_size)
    dataset = dataset.prefetch(10)
    return dataset
