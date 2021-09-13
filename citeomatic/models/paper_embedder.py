import logging

from keras.engine import Model
from keras.layers import Add

from citeomatic.models.layers import L2Normalize, ScalarMultiply, custom_dot
from citeomatic.models.options import ModelOptions
from citeomatic.models.text_embeddings import _prefix, make_embedder

FIELDS = ['title', 'abstract']
SOURCE_NAMES = ['query', 'candidate']


def create_model(options: ModelOptions, pretrained_embeddings=None):
    logging.info('Building model: %s' % options)

    scalar_sum_models = {}
    for field in FIELDS:
        scalar_sum_models[field] = ScalarMultiply(name='scalar-mult-' + field)  # 一个缩放层，只有一个缩放权值参数w


    # same embedders for query and for candidate
    # todo train-4 获得主题嵌入模型和摘要嵌入模型，取sum时两者相同
    embedder_title, embedder_abstract = make_embedder(options, pretrained_embeddings)
    embedders = {'title': embedder_title, 'abstract': embedder_abstract}

    # apply text embedding model and add up title, abstract, etc
    embedding_models = {}
    normed_weighted_sum_of_normed_sums = {}
    for source in SOURCE_NAMES:
        weighted_normed_sums = []
        for field in FIELDS:
            prefix = _prefix((source, field))
            embedding_model, _ = embedders[field].create_text_embedding_model(
                prefix=prefix, final_l2_norm=True
            )
            embedding_models[prefix] = embedding_model
            normed_sum = embedding_models[prefix].outputs[0]
            weighted_normed_sums.append(scalar_sum_models[field](normed_sum))  # 对原始的title向量和论文abstract向量缩放

        weighted_sum = Add()(weighted_normed_sums)  # 将归一化的论文title向量和论文abstract向量相加，得到最终的paper嵌入
        normed_weighted_sum_of_normed_sums[source] = L2Normalize.invoke(
            weighted_sum, name='%s-l2_normed_sum' % source
        )

    # cos distance
    text_output = custom_dot(
        normed_weighted_sum_of_normed_sums['query'],
        normed_weighted_sum_of_normed_sums['candidate'],
        options.dense_dim,
        normalize=False
    )  # 计算两个向量的cos相似度

    citeomatic_inputs = []
    for source in SOURCE_NAMES:
        for field in FIELDS:
            key = _prefix((source, field))
            citeomatic_inputs.append(embedding_models[key].input)

    citeomatic_model = Model(inputs=citeomatic_inputs, outputs=text_output)

    # FIXME 此处可能写错了，len(SOURCE_NAMES)应该为len(FIELDS)，但由于当前两个值相同，所以结果相同
    embedding_model = Model(
        inputs=citeomatic_inputs[0:len(SOURCE_NAMES)],
        outputs=normed_weighted_sum_of_normed_sums['query']
    )

    models = {'embedding': embedding_model, 'citeomatic': citeomatic_model}

    return models
