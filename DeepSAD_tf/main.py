import argparse
import logging
import os
import tensorflow as tf
from tensorflow.keras.datasets import mnist

from datasets.main import load_dataset 
from utils import load_yaml, set_memory_growth
from deepsad import DeepSAD

os.environ['CUDA_VISIBLE_DEVICES']='3'
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, default='./configs/deep_sad_pipeline_audio.yaml', help='config_path')
    
    return parser.parse_args()
    
    
def main():
    args = parse_args()
    
    # Get configuration
    cfg = load_yaml(args.cfg_path)
    
    # Set up log
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    set_memory_growth()
    
    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    
    # define network 
    model = DeepSAD(cfg)
    model.build_model()
    
    # load train & test set
    train_dataset, test_dataset = load_dataset(cfg)
    
    # pretrain
    if cfg['pretrain']:
        model.pretrain(train_dataset, cfg['ae_lr'], cfg['ae_epochs'])
    
    # train
    if cfg['train']:
        model.train(train_dataset, cfg['lr'], cfg['epochs'])
        model.save_model(model_name='saved_models/SAD_model_03_06_3')
    
    # test
    model.test(test_dataset)


if __name__ == '__main__':
    main()