import sys

sys.path.append('/home/daim_gpu/sy/AlphaML')

from alphaml.engine.components.feature_engineering_operator.unary import *
from alphaml.engine.components.feature_engineering.lfe import *


def test():
    test_save()


def test_save():
    import os
    import pickle as pkl
    with open(os.path.join('lfe/data', 'qsa_madelon'), 'rb') as f:
        qsa = pkl.load(f)
    with open(os.path.join('lfe/data', 'label_madelon'), 'rb') as f:
        label = pkl.load(f)
        from collections import Counter
        for i in label:
            print(i,Counter(label[i]))
    # print(qsa, label)


def test_operator():
    col = [-1, 1, 2, 3, -4, 4, 4, 5, 6, 100]
    logop = LogOperator()
    sqrtop = SqrtOperator()
    sqop = SquareOperator()
    fop = FreqOperator()
    rop = RoundOperator()
    tanhop = TanhOperator()
    sigop = SigmoidOperator()
    irop = IsotonicOperator()
    zsop = ZscoreOperator()
    nmop = NormalizeOperator()
    print(logop.operate(col))
    print(sqrtop.operate(col))
    print(sqop.operate(col))
    print(fop.operate(col))
    print(rop.operate(col))
    print(tanhop.operate(col))
    print(sigop.operate(col))
    print(irop.operate(col))
    print(zsop.operate(col))
    print(nmop.operate(col))


def test_valid():
    from alphaml.datasets.cls_dataset.dataset_loader import load_data

    lfe = LFE()
    x, y, _ = load_data('pc4')
    x = np.array(x, dtype=float)
    print(valid_sample(x, y, 0))


def test_generate():
    from alphaml.datasets.cls_dataset.dataset_loader import load_data

    lfe = LFE()
    x, y, _ = load_data('pc4')
    x = np.array(x, dtype=float)
    x = x[:, 0]
    result = lfe.generate_qsa(x, y)
    print(result.shape)
    print(result)


def test_lfe():
    from alphaml.datasets.cls_dataset.dataset_loader import load_data
    lfe = LFE()
    dataset_name = 'madelon'
    x, y, _ = load_data(dataset_name)
    x = np.array(x, dtype=float)
    qsa, dict = lfe.generate_samples(x, y, dataset_name)
    print(qsa, dict)
    print(qsa.shape)


test()
