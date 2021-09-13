import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Lambda, Embedding
from keras.layers import Concatenate, Dot, Reshape, Flatten


class EmbeddingZero(Embedding):
    def call(self, inputs):
        # inputs应为整数序列，每个整数对应一个词
        if K.dtype(inputs) != 'int32':
            inputs = K.cast(inputs, 'int32')
        out = K.gather(self.embeddings, inputs)  # 从权重中选择嵌入输出，2D tensor
        mask = K.expand_dims(K.clip(K.cast(inputs, 'float32'), 0, 1), axis=-1) # 从(n,) 转为（n,1)
        # 由于0表示的单词为占位符，没有对应单词，所以需要将0表示的嵌入通过mask全部清0
        return out * mask


# 给Lambda层命名
class NamedLambda(Lambda):
    def __init__(self, name=None):
        Lambda.__init__(self, self.fn, name=name)

    @classmethod
    def invoke(cls, args, **kw):
        return cls(**kw)(args)

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, self.name)

# L2归一化层，对应公式中的二范数的除法归一化步骤
class L2Normalize(NamedLambda):
    def fn(self, x):
        return K.l2_normalize(x, axis=-1)  # axis=-1表示以最后一维的二范数求归一化，即2d数组每一行作为一个向量，


class ScalarMul(NamedLambda):
    # 参数x包含两个向量，x[0]为嵌入向量2D数组，x[1]为magnitude的1D列向量，计算两者的基本积（不是矩阵乘法）
    def fn(self, x):
        return x[0] * x[1]


class Sum(NamedLambda):
    # 按照第1维相加
    def fn(self, x):
        return K.sum(x, axis=1)


class ScalarMultiply(Layer):
    # 一个缩放层，只有一个缩放权值参数w
    def __init__(self, **kwargs):
        super(ScalarMultiply, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.w = self.add_weight(
            shape=(1, 1), initializer='one', trainable=True, name='w'
        )
        super(ScalarMultiply, self).build(input_shape)

    def call(self, x, mask=None):
        return self.w * x

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1]


def custom_dot(a, b, d, normalize=True):
    # keras is terrible...
    reshaped_a = Reshape((1, d))(a)
    reshaped_b = Reshape((1, d))(b)
    # reshaped_a和b的shape为（None,1,d）
    reshaped_in = [reshaped_a, reshaped_b]
    dotted = Dot(axes=(2, 2), normalize=normalize)(reshaped_in)
    # normalize为true表示进行L2归一化，结果为cos，如果输入已经归一化过了，则不需要
    # dotted的shape为（None,1,1）
    return Flatten()(dotted)


def triplet_loss(y_true, y_pred):
    y_pred = K.flatten(y_pred)
    y_true = K.flatten(y_true)
    pos = y_pred[::2]
    neg = y_pred[1::2]
    # margin is given by the difference in labels
    margin = y_true[::2] - y_true[1::2]
    delta = K.maximum(margin + neg - pos, 0)
    return K.mean(delta, axis=-1)
