data_dir_template = 'data/cls_data/%s/'


def load_data(dataset_name):
    if dataset_name == 'iris':
        from sklearn.datasets import load_iris
        iris = load_iris()
        X, y = iris.data, iris.target
        num_cls = 3
    elif dataset_name == 'digits':
        from sklearn.datasets import load_digits
        digits = load_digits()
        X, y = digits.data, digits.target
        num_cls = 10
    elif dataset_name == 'wine':
        from sklearn.datasets import load_wine
        wine = load_wine()
        X, y = wine.data, wine.target
        num_cls = 3
    elif dataset_name == 'breast_cancer':
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        X, y = data.data, data.target
        num_cls = 2
    elif dataset_name == 'fall_detection':
        from alphaml.datasets.cls_dataset.fall_detection import load_fall_detection
        X, y = load_fall_detection(data_dir_template % dataset_name)
        num_cls = 6
    elif dataset_name == 'dermatology':
        from alphaml.datasets.cls_dataset.dermatology import load_dermatology
        X, y = load_dermatology(data_dir_template % dataset_name)
        num_cls = 4
    elif dataset_name == 'protein':
        from alphaml.datasets.cls_dataset.protein import load_protein
        X, y = load_protein(data_dir_template % dataset_name)
        num_cls = 3
    elif dataset_name == 'australian_scale':
        from alphaml.datasets.cls_dataset.australian import load_australian
        X, y = load_australian(data_dir_template % dataset_name)
        num_cls = 2
    elif dataset_name == 'sensitvehicle':
        from alphaml.datasets.cls_dataset.sensit_vehicle import load_sensit_vehicle
        X, y = load_sensit_vehicle(data_dir_template % dataset_name)
        num_cls = 7
    elif dataset_name == 'connect4':
        from alphaml.datasets.cls_dataset.connect_4 import load_connect_4
        X, y = load_connect_4(data_dir_template % dataset_name)
        num_cls = 3
    elif dataset_name == 'epsilon':
        from alphaml.datasets.cls_dataset.epsilon import load_epsilon
        X, y = load_epsilon(data_dir_template % dataset_name)
        num_cls = 2
    elif dataset_name == 'letter':
        from alphaml.datasets.cls_dataset.letter import load_letter
        X, y = load_letter(data_dir_template % dataset_name)
        num_cls = 26
    elif dataset_name == 'glass':
        from alphaml.datasets.cls_dataset.glass import load_glass
        X, y = load_glass(data_dir_template % dataset_name)
        num_cls = 6
    elif dataset_name == 'satimage':
        from alphaml.datasets.cls_dataset.satimage import load_satimage
        X, y = load_satimage(data_dir_template % dataset_name)
        num_cls = 6
    elif dataset_name == 'svmguide4':
        from alphaml.datasets.cls_dataset.svm_guide4 import load_svmguide4
        X, y = load_svmguide4(data_dir_template % dataset_name)
        num_cls = 6
    elif dataset_name == 'svmguide2':
        from alphaml.datasets.cls_dataset.svm_guide2 import load_svmguide2
        X, y = load_svmguide2(data_dir_template % dataset_name)
        num_cls = 3
    elif dataset_name.startswith('synthetic'):
        from alphaml.datasets.cls_dataset.synthetic import load_synthetic
        id = int(dataset_name[-1])
        X, y = load_synthetic(data_dir_template % 'synthetic', id)
        if id == 0:
            num_cls = 2
        elif id == 1:
            num_cls = 6
        elif id == 2:
            num_cls = 10
        else:
            raise ValueError('Invalid id: %d' % id)

    # Open ML datasets.
    elif dataset_name == 'optdigits':
        from alphaml.datasets.cls_dataset.optdigits import load_optdigits
        X, y = load_optdigits(data_dir_template % dataset_name)
        num_cls = 10
    elif dataset_name == 'usps':
        from alphaml.datasets.cls_dataset.usps import load_usps
        X, y = load_usps(data_dir_template % dataset_name)
        num_cls = 10
    elif dataset_name == 'musk':
        from alphaml.datasets.cls_dataset.musk import load_musk
        X, y = load_musk(data_dir_template % dataset_name)
        num_cls = 2
    elif dataset_name == 'letter':
        from alphaml.datasets.cls_dataset.letter import load_letter
        X, y = load_letter(data_dir_template % dataset_name)
        num_cls = 26
    elif dataset_name == 'a9a':
        from alphaml.datasets.cls_dataset.a9a import load_a9a
        X, y = load_a9a(data_dir_template % dataset_name)
        num_cls = 2
    elif dataset_name == 'pc4':
        from alphaml.datasets.cls_dataset.pc4 import load_pc4
        X, y = load_pc4(data_dir_template % dataset_name)
        num_cls = 2
    elif dataset_name == 'dna':
        from alphaml.datasets.cls_dataset.dna import load_dna
        X, y = load_dna(data_dir_template % dataset_name)
        num_cls = 3
    elif dataset_name == 'segment':
        from alphaml.datasets.cls_dataset.segment import load_segment
        X, y = load_segment(data_dir_template % dataset_name)
        num_cls = 7
    elif dataset_name == 'pendigits':
        from alphaml.datasets.cls_dataset.pendigits import load_pendigits
        X, y = load_pendigits(data_dir_template % dataset_name)
        num_cls = 10
    elif dataset_name == 'satimage':
        from alphaml.datasets.cls_dataset.satimage import load_satimage
        X, y = load_satimage(data_dir_template % dataset_name)
        num_cls = 6
    elif dataset_name == 'quake':
        from alphaml.datasets.cls_dataset.quake import load_quake
        X, y = load_quake(data_dir_template % dataset_name)
        num_cls = 2
    elif dataset_name == 'splice':
        from alphaml.datasets.cls_dataset.splice import load_splice
        X, y = load_splice(data_dir_template % dataset_name)
        num_cls = 3
    elif dataset_name == 'kropt':
        from alphaml.datasets.cls_dataset.kropt import load_kropt
        X, y = load_kropt(data_dir_template % dataset_name)
        num_cls = 18
    elif dataset_name == 'mnist':
        from alphaml.datasets.cls_dataset.mnist import load_mnist
        X, y = load_mnist(data_dir_template % dataset_name)
        num_cls = 10
    elif dataset_name == 'fri_c1':
        from alphaml.datasets.cls_dataset.fri_c1 import load_fri_c1
        X, y = load_fri_c1(data_dir_template % dataset_name)
        num_cls = 2
    elif dataset_name == 'magic_telescope':
        from alphaml.datasets.cls_dataset.magic_telescope import load_magic_telescope
        X, y = load_magic_telescope(data_dir_template % dataset_name)
        num_cls = 2
    elif dataset_name == 'fbis_wc':
        from alphaml.datasets.cls_dataset.fbis_wc import load_fbis_wc
        X, y = load_fbis_wc(data_dir_template % dataset_name)
        num_cls = 17
    elif dataset_name == 'gisette':
        from alphaml.datasets.cls_dataset.gisette import load_gisette
        X, y = load_gisette(data_dir_template % dataset_name)
        num_cls = 2
    elif dataset_name == 'ova_breast':
        from alphaml.datasets.cls_dataset.ova_breast import load_ova_breast
        X, y = load_ova_breast(data_dir_template % dataset_name)
        num_cls = 2
    elif dataset_name == 'shuttle':
        from alphaml.datasets.cls_dataset.shuttle import load_shuttle
        X, y = load_shuttle(data_dir_template % dataset_name)
        num_cls = 7
    elif dataset_name == 'poker':
        from alphaml.datasets.cls_dataset.poker import load_poker
        X, y = load_poker(data_dir_template % dataset_name)
        num_cls = 10
    elif dataset_name == 'waveform':
        from alphaml.datasets.cls_dataset.waveform import load_waveform
        X, y = load_waveform(data_dir_template % dataset_name)
        num_cls = 3
    elif dataset_name == 'madelon':
        from alphaml.datasets.cls_dataset.madelon import load_madelon
        X, y = load_madelon(data_dir_template % dataset_name)
        num_cls = 2
    elif dataset_name == 'semeion':
        from alphaml.datasets.cls_dataset.semeion import load_semeion
        X, y = load_semeion(data_dir_template % dataset_name)
        num_cls = 10
    elif dataset_name == 'wine_quality':
        from alphaml.datasets.cls_dataset.wine_quality import load_wine_quality
        X, y = load_wine_quality(data_dir_template % dataset_name)
        num_cls = 6
    elif dataset_name == 'cifar10s':
        from alphaml.datasets.cls_dataset.cifar10small import load_cifar10s
        X, y = load_cifar10s(data_dir_template % dataset_name)
        num_cls = 10
    elif dataset_name == 'eucalyptus':
        from alphaml.datasets.cls_dataset.eucalyptus import load_eucalyptus
        X, y = load_eucalyptus(data_dir_template % dataset_name)
        num_cls = 5
    elif dataset_name == 'yeast':
        from alphaml.datasets.cls_dataset.yeast import load_yeast
        X, y = load_yeast(data_dir_template % dataset_name)
        num_cls = 10
    elif dataset_name == 'wind':
        from alphaml.datasets.cls_dataset.wind import load_wind
        X, y = load_wind(data_dir_template % dataset_name)
        num_cls = 2
    elif dataset_name == 'vehicle':
        from alphaml.datasets.cls_dataset.vehicle import load_vehicle
        X, y = load_vehicle(data_dir_template % dataset_name)
        num_cls = 4
    elif dataset_name == 'mammography':
        from alphaml.datasets.cls_dataset.mammography import load_mammography
        X, y = load_mammography(data_dir_template % dataset_name)
        num_cls = 2
    elif dataset_name == 'houses':
        from alphaml.datasets.cls_dataset.houses import load_houses
        X, y = load_houses(data_dir_template % dataset_name)
        num_cls = 2
    elif dataset_name == 'elevators':
        from alphaml.datasets.cls_dataset.elevators import load_elevators
        X, y = load_elevators(data_dir_template % dataset_name)
        num_cls = 2
    elif dataset_name == 'electricity':
        from alphaml.datasets.cls_dataset.electricity import load_electricity
        X, y = load_electricity(data_dir_template % dataset_name)
        num_cls = 2
    elif dataset_name == 'eeg_eye_state':
        from alphaml.datasets.cls_dataset.eeg_eye_state import load_eeg_eye_state
        X, y = load_eeg_eye_state(data_dir_template % dataset_name)
        num_cls = 2
    elif dataset_name == 'abalone':
        from alphaml.datasets.cls_dataset.abalone import load_abalone
        X, y = load_abalone(data_dir_template % dataset_name)
        num_cls = 3
    elif dataset_name == 'amazon':
        from alphaml.datasets.cls_dataset.amazon import load_amazon
        X, y = load_amazon(data_dir_template % dataset_name)
        num_cls = 50
    elif dataset_name == 'higgs':
        from alphaml.datasets.cls_dataset.higgs import load_higgs
        X, y = load_higgs(data_dir_template % dataset_name)
        num_cls = 2
    else:
        raise ValueError('Invalid dataset name: %s!' % dataset_name)
    print(X.shape, y.shape)
    print(min(y), max(y), num_cls)
    return X, y, num_cls
