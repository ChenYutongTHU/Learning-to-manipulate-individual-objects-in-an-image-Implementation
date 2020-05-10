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
"""Objects Room dataset reader."""

import functools
import tensorflow as tf


COMPRESSION_TYPE = tf.io.TFRecordOptions.get_compression_type_string('GZIP')
IMAGE_SIZE = [64, 64]
# The maximum number of foreground and background entities in each variant
# of the provided datasets. The values correspond to the number of
# segmentation masks returned per scene.
MAX_NUM_ENTITIES = {
    'train': 7,
    'six_objects': 10,
    'empty_room': 4,
    'identical_color': 10
}
BYTE_FEATURES = ['mask', 'image']


def feature_descriptions(max_num_entities):
  """Create a dictionary describing the dataset features.
  Args:
    max_num_entities: int. The maximum number of foreground and background
      entities in each image. This corresponds to the number of segmentation
      masks returned per scene.
  Returns:
    A dictionary which maps feature names to `tf.Example`-compatible shape and
    data type descriptors.
  """
  return {
      'image': tf.FixedLenFeature(IMAGE_SIZE+[3], tf.string),
      'mask': tf.FixedLenFeature([max_num_entities]+IMAGE_SIZE+[1], tf.string),
  }


def _decode(example_proto, features, random_sky):
  # Parse the input `tf.Example` proto using a feature description dictionary.
  single_example = tf.parse_single_example(example_proto, features)
  for k in BYTE_FEATURES:
    single_example[k] = tf.squeeze(tf.decode_raw(single_example[k], tf.uint8),
                                   axis=-1)
  # sky floor half-wall1(2) half-wall2(3) objects objects objects
  mask = tf.transpose(single_example['mask'], [1, 2, 3, 0]) #H W 1 7
  mask = tf.concat([mask[:,:,:,0:2],mask[:,:,:,2:3]+mask[:,:,:,3:4],mask[:,:,:,4:]], axis=-1) #H W 1 6 merge the wall
  single_example['mask'] = mask
  return map(single_example, random_sky)


def map(x, random_sky):
  img = tf.cast(x['image']/255, tf.float32) #0~1
  mask = tf.cast(x['mask']/255, tf.float32)

  data = {}
  data['masks'] = mask

  sky_mask = mask[:,:,:,0]#H W 1
  scale = tf.random_uniform(shape=[], minval=0.2, maxval=1, dtype=tf.float32) if random_sky else 1
  var_img = img*(1-sky_mask)+img*scale*sky_mask #0~1
  data['img'] = var_img
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

  max_num_entities = MAX_NUM_ENTITIES['train']
  raw_dataset = tf.data.TFRecordDataset(
      tfrecords_path, compression_type=COMPRESSION_TYPE,
      buffer_size=50, num_parallel_reads=2)
  features = feature_descriptions(max_num_entities)
  partial_decode_fn = functools.partial(_decode, features=features, random_sky=(phase=='train')) #val constant sky

  dataset = raw_dataset.skip(skipnum).take(takenum)
  dataset = dataset.map(partial_decode_fn, num_parallel_calls=1)
  if shuffle:
      dataset = dataset.shuffle(seed=479, buffer_size=batch_size*100, reshuffle_each_iteration=True)
  dataset = dataset.repeat().batch(batch_size)
  dataset = dataset.prefetch(10)
  return dataset


# import imageio
# import numpy as np
# bs = 4
# dataset = dataset('./objects_room_data/objects_room_train.tfrecords',val=True,
#                 batch_size=bs, skipnum=0, takenum=-1,
#                 shuffle=False, map_parallel_calls=1)

# iterator = dataset.make_one_shot_iterator()

# data_batch = iterator.get_next()
# edge_batch = bin_edge_map(data_batch['img'], 'objects_room')

# sess = tf.Session()
# for i in range(3):
#   data, edges = sess.run((data_batch, edge_batch))
#   for k in range(bs):
#     img = data['img'][k,:,:,:]
#     masks = data['masks'][k,:,:,:,:]
#     imageio.imwrite('debug/{}_{}img.png'.format(i,k), (img*255).astype(np.uint8))

#     edge = edges[k,:,:,:] #H W 2
#     show = np.concatenate([edge, np.zeros_like(edge[:,:,0:1])], axis=-1)#H W 3
#     imageio.imwrite('debug/{}_{}edge.png'.format(i,k), (show*255).astype(np.uint8))
    # for m in range(6):
    #   imageio.imwrite('debug/{}_{}mask{}.png'.format(i,k,m), (masks[:,:,:,m]*img*255).astype(np.uint8))