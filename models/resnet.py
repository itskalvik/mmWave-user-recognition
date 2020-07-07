import tensorflow as tf

L2_WEIGHT_DECAY = 1e-4
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5
"""A block that has a identity layer at shortcut.
Args:
  kernel_size: the kernel size of middle conv layer at main path
  filters: list of integers, the filters of 3 conv layer at main path
  stage: integer, current stage label, used for generating layer names
  block: 'a','b'..., current block label, used for generating layer names
  activation: activation function to use in all layers in the block

Returns:
  A Keras model instance for the block.
"""


class IdentityBlock(tf.keras.Model):
    def __init__(self,
                 kernel_size,
                 filters,
                 stage,
                 block,
                 activation='relu',
                 regularizer='batchnorm',
                 dropout_rate=0):
        self.activation = activation

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        super().__init__(name='stage-' + str(stage) + '_block-' + block)

        filters1, filters2 = filters
        bn_axis = -1

        self.conv2a = tf.keras.layers.Conv2D(
            filters1,
            kernel_size,
            padding='same',
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
            name=conv_name_base + '2a')
        if regularizer.lower() == 'dropout':
            self.bn2a = tf.keras.layers.Dropout(rate=dropout_rate,
                                                name=bn_name_base + '2a')
        elif regularizer.lower() == 'batchnorm':
            self.bn2a = tf.keras.layers.BatchNormalization(
                axis=bn_axis,
                momentum=BATCH_NORM_DECAY,
                epsilon=BATCH_NORM_EPSILON,
                name=bn_name_base + '2a')
        self.act1 = tf.keras.layers.Activation(self.activation)

        self.conv2b = tf.keras.layers.Conv2D(
            filters2,
            kernel_size,
            padding='same',
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
            name=conv_name_base + '2b')
        if regularizer.lower() == 'dropout':
            self.bn2b = tf.keras.layers.Dropout(rate=dropout_rate,
                                                name=bn_name_base + '2b')
        elif regularizer.lower() == 'batchnorm':
            self.bn2b = tf.keras.layers.BatchNormalization(
                axis=bn_axis,
                momentum=BATCH_NORM_DECAY,
                epsilon=BATCH_NORM_EPSILON,
                name=bn_name_base + '2b')
        self.act2 = tf.keras.layers.Activation(self.activation)

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = self.act1(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)

        x = tf.keras.layers.add([x, input_tensor])
        x = self.act2(x)
        return x


"""A block that has a conv layer at shortcut.

Note that from stage 3,
the second conv layer at main path is with strides=(2, 2)
And the shortcut should have strides=(2, 2) as well

Args:
  kernel_size: the kernel size of middle conv layer at main path
  filters: list of integers, the filters of 3 conv layer at main path
  stage: integer, current stage label, used for generating layer names
  block: 'a','b'..., current block label, used for generating layer names
  strides: Strides for the second conv layer in the block.
  activation: activation function to use in all layers in the block

Returns:
  A Keras model instance for the block.
"""


class ConvBlock(tf.keras.Model):
    def __init__(self,
                 kernel_size,
                 filters,
                 stage,
                 block,
                 strides=(2, 2),
                 activation='relu',
                 regularizer='batchnorm',
                 dropout_rate=0):
        self.activation = activation

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        super().__init__(name='stage-' + str(stage) + '_block-' + block)

        filters1, filters2 = filters
        bn_axis = -1

        self.conv2a = tf.keras.layers.Conv2D(
            filters1,
            kernel_size,
            padding='same',
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
            name=conv_name_base + '2a')
        if regularizer.lower() == 'dropout':
            self.bn2a = tf.keras.layers.Dropout(rate=dropout_rate,
                                                name=bn_name_base + '2a')
        elif regularizer.lower() == 'batchnorm':
            self.bn2a = tf.keras.layers.BatchNormalization(
                axis=bn_axis,
                momentum=BATCH_NORM_DECAY,
                epsilon=BATCH_NORM_EPSILON,
                name=bn_name_base + '2a')
        self.act1 = tf.keras.layers.Activation(self.activation)

        self.conv2b = tf.keras.layers.Conv2D(
            filters2,
            kernel_size,
            strides=strides,
            padding='same',
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
            name=conv_name_base + '2b')
        if regularizer.lower() == 'dropout':
            self.bn2b = tf.keras.layers.Dropout(rate=dropout_rate,
                                                name=bn_name_base + '2b')
        elif regularizer.lower() == 'batchnorm':
            self.bn2b = tf.keras.layers.BatchNormalization(
                axis=bn_axis,
                momentum=BATCH_NORM_DECAY,
                epsilon=BATCH_NORM_EPSILON,
                name=bn_name_base + '2b')
        self.act2 = tf.keras.layers.Activation(self.activation)

        self.conv2s = tf.keras.layers.Conv2D(
            filters2,
            kernel_size,
            strides=strides,
            padding='same',
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
            name=conv_name_base + '1')
        if regularizer.lower() == 'dropout':
            self.bn2s = tf.keras.layers.Dropout(rate=dropout_rate,
                                                name=bn_name_base + '2s')
        elif regularizer.lower() == 'batchnorm':
            self.bn2s = tf.keras.layers.BatchNormalization(
                axis=bn_axis,
                momentum=BATCH_NORM_DECAY,
                epsilon=BATCH_NORM_EPSILON,
                name=bn_name_base + '2s')

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = self.act1(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)

        shortcut = self.conv2s(input_tensor)
        shortcut = self.bn2s(shortcut, training=training)

        x = tf.keras.layers.add([x, shortcut])
        x = self.act2(x)
        return x


"""Instantiates the ResNet50 architecture.

Args:
  num_classes: `int` number of classes for image classification.

Returns:
    A Keras model instance.
"""


class ResNet50(tf.keras.Model):
    def __init__(self,
                 num_classes,
                 num_features,
                 num_filters=64,
                 activation='relu',
                 regularizer='batchnorm',
                 dropout_rate=0):
        super().__init__(name='generator')
        bn_axis = -1
        self.activation = activation
        self.num_classes = num_classes

        self.conv1 = tf.keras.layers.Conv2D(
            num_filters, (7, 7),
            strides=(2, 2),
            padding='same',
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
            name='conv1')
        if regularizer.lower() == 'dropout':
            self.bn1 = tf.keras.layers.Dropout(rate=dropout_rate,
                                               name='bn_conv1')
        elif regularizer.lower() == 'batchnorm':
            self.bn1 = tf.keras.layers.BatchNormalization(
                axis=bn_axis,
                momentum=BATCH_NORM_DECAY,
                epsilon=BATCH_NORM_EPSILON,
                name='bn_conv1')
        self.act1 = tf.keras.layers.Activation(self.activation,
                                               name=self.activation + '1')
        self.max_pool1 = tf.keras.layers.MaxPooling2D((3, 3),
                                                      strides=(2, 2),
                                                      padding='same',
                                                      name='max_pool1')

        self.blocks = []
        self.blocks.append(
            ConvBlock(3, [num_filters, num_filters],
                      strides=(1, 1),
                      stage=2,
                      block='a',
                      activation=self.activation,
                      regularizer=regularizer,
                      dropout_rate=dropout_rate))
        self.blocks.append(
            IdentityBlock(3, [num_filters, num_filters],
                          stage=2,
                          block='b',
                          activation=self.activation,
                          regularizer=regularizer,
                          dropout_rate=dropout_rate))

        self.blocks.append(
            ConvBlock(3, [num_filters * 2, num_filters * 2],
                      stage=3,
                      block='a',
                      activation=self.activation,
                      regularizer=regularizer,
                      dropout_rate=dropout_rate))
        self.blocks.append(
            IdentityBlock(3, [num_filters * 2, num_filters * 2],
                          stage=3,
                          block='b',
                          activation=self.activation,
                          regularizer=regularizer,
                          dropout_rate=dropout_rate))

        self.blocks.append(
            ConvBlock(3, [num_filters * 4, num_filters * 4],
                      stage=4,
                      block='a',
                      activation=self.activation,
                      regularizer=regularizer,
                      dropout_rate=dropout_rate))
        self.blocks.append(
            IdentityBlock(3, [num_filters * 4, num_filters * 4],
                          stage=4,
                          block='b',
                          activation=self.activation,
                          regularizer=regularizer,
                          dropout_rate=dropout_rate))

        self.blocks.append(
            ConvBlock(3, [num_filters * 8, num_filters * 8],
                      stage=5,
                      block='a',
                      activation=self.activation,
                      regularizer=regularizer,
                      dropout_rate=dropout_rate))
        self.blocks.append(
            IdentityBlock(3, [num_filters * 8, num_filters * 8],
                          stage=5,
                          block='b',
                          activation=self.activation,
                          regularizer=regularizer,
                          dropout_rate=dropout_rate))

        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')
        self.fc1 = tf.keras.layers.Dense(
            num_features,
            activation=self.activation,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
            bias_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
            name='fc1')

        self.logits = tf.keras.layers.Dense(
            num_classes,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
            bias_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
            name='logits')

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.max_pool1(x)

        for block in self.blocks:
            x = block(x, training=training)

        x = self.avg_pool(x)
        fc1 = self.fc1(x)
        logits = self.logits(fc1)

        return logits, fc1
