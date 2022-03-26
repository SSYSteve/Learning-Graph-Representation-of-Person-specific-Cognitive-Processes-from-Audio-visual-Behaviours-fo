from collections import namedtuple

# Genotype = namedtuple('Genotype', 'Modal_1 Modal_2 Modal_f')

PRIMITIVES = [
    # 'none',
    'max_pool_3',
    'avg_pool_3',
    'skip_connect',
    'sep_conv_3',
    'sep_conv_5',
    'dil_conv_3',
    'dil_conv_5'
]

PRIMITIVES_DOWN = [
    'max_pool_3',
    'avg_pool_3',
    'sep_conv_3',
    'sep_conv_5',
    'dil_conv_3',
    'dil_conv_5'
]

PRIMITIVES_UP = [
    'tran_conv_3',
    'linear',
    'nearest'
]

# TESTG = Genotype(Modal_1={'Down': [('max_pool_3',), ('avg_pool_3',)], 'NotDown': [('max_pool_3',), ('sep_conv_3',)], 'DownStep': [('avg_pool_3', 0), ('sep_conv_3', 0), ('skip_connect', 1), ('skip_connect', 1), ('skip_connect', 0)], 'Up': [('tran_conv_3',), ('linear',)], 'NotUp': [('dil_conv_5',), ('dil_conv_3',)], 'UpStep': [('skip_connect', 0),('skip_connect', 0), ('dil_conv_5', 1), ('avg_pool_3', 0), ('dil_conv_3', 1), ('dil_conv_3', 0)]}, Modal_2={'Down': [('avg_pool_3',), ('dil_conv_3',)], 'NotDown': [('dil_conv_5',), ('max_pool_3',)], 'DownStep': [('sep_conv_3', 0), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 1), ('skip_connect', 2)], 'Up': [('nearest',), ('nearest',)], 'NotUp': [('avg_pool_3',), ('dil_conv_5',)], 'UpStep': [('sep_conv_3', 0), ('sep_conv_3', 0),('skip_connect', 1), ('sep_conv_5', 0), ('dil_conv_3', 2), ('sep_conv_3', 0)]}, Modal_f={'Down': [('avg_pool_3',), ('sep_conv_3',), ('dil_conv_3',), ('max_pool_3',)], 'NotDown': [('dil_conv_3',), ('dil_conv_3',), ('dil_conv_3',), ('dil_conv_3',)], 'DownStep': [('avg_pool_3', 1), ('dil_conv_3', 0), ('sep_conv_5', 0), ('dil_conv_5', 2), ('dil_conv_3', 0), ('avg_pool_3', 1)], 'Up': [('tran_conv_3',), ('tran_conv_3',), ('linear',), ('tran_conv_3',)], 'NotUp': [('dil_conv_3',), ('avg_pool_3',), ('dil_conv_5',), ('sep_conv_3',)], 'UpStep': [('skip_connect', 1),('skip_connect', 0), ('sep_conv_3', 0), ('skip_connect', 2), ('skip_connect', 1), ('sep_conv_5', 3), ('avg_pool_3', 0)]})
# TESTG_2 = Genotype(Modal_1={'Down': [('dil_conv_3',), ('avg_pool_3',)], 'NotDown': [('max_pool_3',), ('sep_conv_3',)], 'DownStep': [('dil_conv_3', 0), ('sep_conv_3', 0), ('dil_conv_5', 1), ('sep_conv_5', 2), ('dil_conv_3', 1)], 'Up': [('linear',), ('linear',)], 'NotUp': [('dil_conv_5',), ('dil_conv_3',)], 'UpStep': [('dil_conv_3', 0),('sep_conv_3', 1), ('sep_conv_3', 1), ('max_pool_3', 0), ('sep_conv_3', 2), ('skip_connect', 1)]}, Modal_2={'Down': [('max_pool_3',), ('max_pool_3',)], 'NotDown': [('dil_conv_5',), ('max_pool_3',)], 'DownStep': [('max_pool_3', 0), ('max_pool_3', 1), ('max_pool_3', 0), ('dil_conv_3', 1), ('dil_conv_3', 2)], 'Up': [('linear',), ('tran_conv_3',)], 'NotUp': [('avg_pool_3',), ('dil_conv_5',)], 'UpStep': [('sep_conv_5', 0),('sep_conv_3', 1), ('dil_conv_5', 1), ('skip_connect', 0), ('dil_conv_3', 0), ('dil_conv_3', 1)]}, Modal_f={'Down': [('sep_conv_5',), ('avg_pool_3',), ('avg_pool_3',), ('sep_conv_5',)], 'NotDown': [('dil_conv_3',), ('dil_conv_3',), ('dil_conv_3',), ('dil_conv_3',)], 'DownStep': [('skip_connect', 1), ('dil_conv_3', 0), ('max_pool_3', 1), ('skip_connect', 0), ('sep_conv_5', 2), ('dil_conv_5', 3)], 'Up': [('tran_conv_3',), ('tran_conv_3',), ('tran_conv_3',), ('tran_conv_3',)], 'NotUp': [('dil_conv_3',), ('avg_pool_3',), ('dil_conv_5',), ('sep_conv_3',)], 'UpStep': [('sep_conv_5', 0), ('dil_conv_5', 1), ('max_pool_3', 0), ('dil_conv_3', 2), ('sep_conv_3', 2), ('sep_conv_5', 1)]})

if __name__ == '__main__':
    print(TESTG.Modal_1)