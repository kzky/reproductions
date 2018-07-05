from nnabla.utils.data_iterator import data_iterator_cache


def data_iterator_imagenet(batch_size, cache_dir, rng=None):
    return data_iterator_cache(cache_dir, batch_size, shuffle=True, normalize=False, rng=rng)
