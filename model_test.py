import numpy as np
import tensorflow as tf
from math import ceil
from Deep_SVDD_data import audio_data_to_net
from Deep_SVDD_data_xb import audio_data_to_net_xb
from Deep_SVDD_data_pro import audio_data_to_net_pro
model = tf.keras.models.load_model('D_VGGish2.h5')
c = np.array([ 0.2265381 ,  4.3677077 ,  1.7365365 , -1.9650414 ,  0.68256986,
       -1.6754267 , -3.9105637 , -0.1       , -2.0360956 ,  1.5499107 ,
       -0.77903765, -1.1754923 , -0.92478347,  0.94411784, -1.5329931 ,
       -0.83420116,  4.6767416 ,  1.5778654 ,  5.660471  ,  2.186515  ,
        0.7074727 ,  0.3858902 , -2.893027  , -3.3713753 ,  2.173305  ,
       -3.1413465 ,  2.9831707 , -2.1224608 , -1.1940266 ,  6.220429  ,
       -0.69757   ,  3.635412  , -5.560528  , -1.032272  , -1.6208292 ,
       -1.9517537 , -3.7535172 ,  6.1162167 , -1.3860365 , -3.428485  ,
        0.96109253, -2.287562  ,  5.303622  , -2.0163295 , -0.2041027 ,
        3.7096028 , -2.6209314 , -2.165892  ,  0.44419   , -3.4776664 ,
       -1.9843988 ,  3.3323216 ,  0.1       ,  1.9536213 , -0.67685944,
        2.5541449 , -0.45630977,  1.4368159 , -1.6018002 , -2.8631027 ,
       -1.6360745 ,  2.9277878 ,  1.6466258 ,  1.959357  ], dtype=np.float32)

def jiance(X, batch_size):
    N = X.shape[0]
    BS = batch_size
    BN = int(ceil(N / BS))
    scores = []

    for i_batch in range(BN):
        x_batch = X[i_batch * BS: (i_batch + 1) * BS]
        s_batch = call(x=x_batch, c=c, R=0, nu=0.1, objective='one-class', deep_model=model)['score']
        scores.append(s_batch)
    print(scores)

def call(x, c, R, nu, objective, deep_model):
    z = deep_model(x)
    dist = tf.reduce_sum(tf.square(z - c), axis=-1)
    if objective == 'soft-boundary':
        score = dist - R ** 2
        loss = R ** 2 + (1 / nu) * tf.maximum(score, tf.zeros_like(score))
    else:
        score = dist
        loss = score
    # self.deep_model.save('D_vggish.h5')
    return {'score': score, 'loss': loss}

if __name__ == '__main__':
    x_test, y_test= audio_data_to_net_pro(excel_dir='testnormalpro/pure_testnormalpro.csv', target_samples=51000, target_sr=22050,
                               root_dir='testnormalpro', hop_size=20000, n_mfcc=20)
    # x_test, y_test= audio_data_to_net_xb(excel_dir='LD_18_test/LD18_test.csv', target_samples=30000, target_sr=22_050,
    #                            root_dir='LD_18_test', hop_size=20000)
    # print(y_test)
    # jiance(x_test, 64)
    svdd = tf.saved_model.load('DeepSAD_tf/saved_models/test')
    print(svdd.c)
    x_test = np.array(x_test)
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float64)
    k = 0
    n = 721 * 4
    thresh_hold = 5
    y_pred = np.zeros_like(y_test)
    for i in range(n):
        a = x_test[i]
        a = np.expand_dims(a, axis=0)
        score = svdd.predict_fcn(a)
        print(i, score, y_test[i])
        # 精度计算
        if score >= thresh_hold:
            y_pred[i] = 1
        else:y_pred[i] = 0
    for i in range(n):
        if int(y_pred[i]) == int(y_test[i]):
            k = k + 1
    acc = k / n * 100
    print('该模型测试精度为：', acc, '%')
