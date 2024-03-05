import json
import tensorflow as tf
import pickle

from base.base_dataset import BaseADDataset
from base.networks_main import build_network, build_autoencoder
from base.DeepSAD_trainer import DeepSADTrainer
from base.ae_trainer import AETrainer


class DeepSAD(object):
    """A class for the Deep SAD method.

    Attributes:
        eta: 深度 SAD 超参数 eta（必须为 0 < eta）.
        c: 超球面中心 c.
        net_name: 一个字符串，指示要使用的神经网络的名称。
        net: 神经网络 phi。
        trainer: DeepSADTrainer 用于训练 Deep SAD 模型。
        optimizer_name: 一个字符串，指示用于训练 Deep SAD 网络的优化器。
        ae_net: 与phi对应的自动编码器网络，用于网络权重预训练。
        ae_trainer: AETrainer 在预训练中训练自动编码器。
        ae_optimizer_name: 指示用于预训练自动编码器的优化器的字符串。
        results: 保存结果的字典。
        ae_results: 保存自动编码器结果的字典。
    """

    def __init__(self, eta: float = 1.0):
        """使用超参数 eta 初始化 DeepSAD"""

        self.eta = eta
        self.c = None  # hypersphere center c

        self.net_name = None
        self.net = None  # neural network phi

        self.trainer = None
        self.optimizer_name = None

        self.ae_net = None  # autoencoder network for pretraining
        self.ae_trainer = None
        self.ae_optimizer_name = None

        self.results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None,
            'test_scores': None,
        }

        self.ae_results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None
        }

    def set_network(self, net_name):
        """构建神经网络 phi"""
        self.net_name = net_name
        self.net = build_network(net_name)

    def train(self, dataset: BaseADDataset, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 50,
              lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
              n_jobs_dataloader: int = 0):
        """在训练数据上训练 Deep SAD 模型"""

        self.optimizer_name = optimizer_name
        self.trainer = DeepSADTrainer(self.c, self.eta, optimizer_name=optimizer_name, lr=lr, n_epochs=n_epochs,
                                      lr_milestones=lr_milestones, batch_size=batch_size, weight_decay=weight_decay,
                                      device=device, n_jobs_dataloader=n_jobs_dataloader)
        # Get the model
        self.net = self.trainer.train(dataset, self.net)
        self.results['train_time'] = self.trainer.train_time
        self.c = self.trainer.c.numpy().tolist()  # get as list

    def test(self, dataset: BaseADDataset, device: str = 'cuda', n_jobs_dataloader: int = 0):
        """在测试数据上测试 Deep SAD 模型"""

        if self.trainer is None:
            self.trainer = DeepSADTrainer(self.c, self.eta, device=device, n_jobs_dataloader=n_jobs_dataloader)

        self.trainer.test(dataset, self.net)

        # Get results
        self.results['test_auc'] = self.trainer.test_auc
        self.results['test_time'] = self.trainer.test_time
        self.results['test_scores'] = self.trainer.test_scores

    def pretrain(self, dataset: BaseADDataset, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 100,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        """通过自动编码器预训练 Deep SAD 网络 phi 的权重"""

        # Set autoencoder network
        self.ae_net = build_autoencoder(self.net_name)

        # Train
        self.ae_optimizer_name = optimizer_name
        self.ae_trainer = AETrainer(optimizer_name, lr=lr, n_epochs=n_epochs, lr_milestones=lr_milestones,
                                    batch_size=batch_size, weight_decay=weight_decay, device=device,
                                    n_jobs_dataloader=n_jobs_dataloader)
        self.ae_net = self.ae_trainer.train(dataset, self.ae_net)

        # Get train results
        self.ae_results['train_time'] = self.ae_trainer.train_time

        # Test
        self.ae_trainer.test(dataset, self.ae_net)

        # Get test results
        self.ae_results['test_auc'] = self.ae_trainer.test_auc
        self.ae_results['test_time'] = self.ae_trainer.test_time

        # Initialize Deep SAD network weights from pre-trained encoder
        self.init_network_weights_from_pretraining()

    def init_network_weights_from_pretraining(self):
        """根据预训练自动编码器的编码器权重初始化 Deep SAD 网络权重"""

        net_weights = self.net.get_weights()
        ae_weights = self.ae_net.get_weights()

        # Filter out decoder network weights
        ae_weights = [ae_w for ae_w, net_w in zip(ae_weights, net_weights)]
        self.net.set_weights(ae_weights)

    def save_model(self, export_model, save_ae=True):
        """将 Deep SAD 模型保存到 export_model"""

        net_weights = self.net.get_weights()
        ae_weights = self.ae_net.get_weights() if save_ae else None

        with open(export_model, 'wb') as model_file:
            model_dict = {
                'c': self.c,
                'net_weights': net_weights,
                'ae_weights': ae_weights
            }
            pickle.dump(model_dict, model_file)

    def load_model(self, model_path, load_ae=False):
        """从 model_path 加载 Deep SAD 模型"""

        with open(model_path, 'rb') as model_file:
            model_dict = pickle.load(model_file)

            self.c = model_dict['c']
            self.net.set_weights(model_dict['net_weights'])

            if load_ae and model_dict['ae_weights'] is not None:
                self.ae_net.set_weights(model_dict['ae_weights'])

    def save_results(self, export_json):
        """将结果字典保存到 JSON 文件"""
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)

    def save_ae_results(self, export_json):
        """将自动编码器结果字典保存到 JSON 文件"""
        with open(export_json, 'w') as fp:
            json.dump(self.ae_results, fp)
