from .mnist import load_mnist
from .cifar10 import load_cifar10
from .pipeline_audio import load_pipeline_audio

def load_dataset(cfg):
    dataset_name = cfg['dataset_name']
    
    implemented_datasets = ('pipeline_audio', 'mnist', 'fmnist', 'cifar10',
                            'arrhythmia', 'cardio', 'satellite', 'satimage-2', 'shuttle', 'thyroid')
    
    assert dataset_name in implemented_datasets
    
    dataset = None
    if dataset_name == 'pipeline_audio':
        dataset = load_pipeline_audio(cfg)
    if dataset_name == 'mnist':
        dataset = load_mnist(cfg)
    if dataset_name == 'cifar10':
        dataset = load_cifar10(cfg)
    
    return dataset