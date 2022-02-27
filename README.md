# tfvan

Keras (TensorFlow v2) reimplementation of **Visual Attention Network** model.
Based on [Official Pytorch implementation](https://github.com/Visual-Attention-Network/VAN-Classification).

Supports variable-shape inference. All weights are obtained by converting official checkpoints. 

## Installation

```bash
pip install tfvan
```

## Examples

Default usage (without preprocessing):

```python
from tfvan import VanTiny  # + 3 other variants and input preprocessing

model = VanTiny()  # by default will download imagenet-pretrained weights
model.compile(...)
model.fit(...)
```

Custom classification (with preprocessing):

```python
from keras import layers, models
from tfvan import VanTiny, preprocess_input

inputs = layers.Input(shape=(224, 224, 3), dtype='uint8')
outputs = layers.Lambda(preprocess_input)(inputs)
outputs = VanTiny(include_top=False)(outputs)
outputs = layers.Dense(100, activation='softmax')(outputs)

model = models.Model(inputs=inputs, outputs=outputs)
model.compile(...)
model.fit(...)
```

## Evaluation

For correctness, `Tiny` and `Small` models (original and ported) tested
with [ImageNet-v2 test set](https://www.tensorflow.org/datasets/catalog/imagenet_v2).

```python
import tensorflow as tf
import tensorflow_datasets as tfds
from tfvan import VanTiny, preprocess_input


def _prepare(example):
    # Observation: +1.3% top1 accuracy in tiny model with antialias=True
    image = tf.image.resize(example['image'], (248, 248), method=tf.image.ResizeMethod.BICUBIC)
    image = tf.image.central_crop(image, 0.9)
    image = preprocess_input(image)

    return image, example['label']


imagenet2 = tfds.load('imagenet_v2', split='test', shuffle_files=True)
imagenet2 = imagenet2.map(_prepare, num_parallel_calls=tf.data.AUTOTUNE)
imagenet2 = imagenet2.batch(8)

model = VanTiny()
model.compile('sgd', 'sparse_categorical_crossentropy', ['accuracy', 'sparse_top_k_categorical_accuracy'])
history = model.evaluate(imagenet2)

print(history)
```

| name | original acc@1 | ported acc@1 | original acc@5 | ported acc@5 |
| :---: | :---: | :---: | :---: | :---: |
| Tiny | 59.22 | 61.59 | 82.32 | 84.52 |
| Small | 70.17 | 68.62 | 89.17 | 88.54 |

## Citation

```
@article{guo2022visual,
  title={Visual Attention Network},
  author={Guo, Meng-Hao and Lu, Cheng-Ze and Liu, Zheng-Ning and Cheng, Ming-Ming and Hu, Shi-Min},
  journal={arXiv preprint arXiv:2202.09741},
  year={2022}
}