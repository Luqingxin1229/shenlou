from base_net import CustomModel_1, MNIST_LeNet_Autoencoder



def build_network(net_name, ae_net=None):
    """Builds the neural network."""

    implemented_networks = ('model1')
    assert net_name in implemented_networks

    net = None

    if net_name == 'model1':
        net = CustomModel_1()

    return net

def build_autoencoder(net_name):
    """Builds the corresponding autoencoder network."""

    implemented_networks = ('model1')

    assert net_name in implemented_networks

    ae_net = None

    if net_name == 'model1':
        ae_net = MNIST_LeNet_Autoencoder()