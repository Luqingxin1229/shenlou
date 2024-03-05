import tensorflow as tf
import numpy as np
import cv2
tf.compat.v1.disable_eager_execution()
# from yanzhengma_train import crack_captcha_cnn_network

# 定义模型结构，确保与保存模型时的结构一致

# 图像尺寸
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160
dictionary1=['0','1','2','3','4','5','6','7','8','9']
def crack_captcha_cnn_network(w_alpha=0.01,b_alpha=0.1):
    x = tf.reshape(X,shape=[-1,IMAGE_HEIGHT,IMAGE_WIDTH,1])

    w_c1 = tf.Variable(w_alpha*tf.compat.v1.random_normal([3,3,1,32]))
    b_c1 = tf.Variable(b_alpha*tf.compat.v1.random_normal([32]))#原代码

    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x,w_c1,strides=[1, 1, 1, 1], padding='SAME'), b_c1))#原代码
    # conv1 = tf.nn.relu(tf.nn.conv2d(x,w_c1, strides=[1, 1, 1, 1], padding='SAME'))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    w_c2 = tf.Variable(w_alpha * tf.compat.v1.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.compat.v1.random_normal([64]))#原代码

    conv2 = tf.nn.elu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))#原代码
    # conv2 = tf.nn.elu(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    w_c3 = tf.Variable(w_alpha * tf.compat.v1.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha * tf.compat.v1.random_normal([64]))#原代码

    conv3 = tf.nn.elu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1],padding='SAME'),b_c3))#原代码
    # conv3 = tf.nn.elu(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'))
    conv3 = tf.nn.max_pool(conv3,ksize=[1,2,2,1],strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3,keep_prob)

    w_d = tf.Variable(w_alpha * tf.compat.v1.random_normal([8*20*64,1024]))
    b_d = tf.Variable(b_alpha * tf.compat.v1.random_normal([1024]))

    dense = tf.reshape(conv3,[-1,w_d.get_shape().as_list()[0]])
    dense = tf.nn.elu(tf.add(tf.matmul(dense,w_d),b_d))
    dense = tf.nn.dropout(dense,keep_prob)

    w_out = tf.Variable(w_alpha*tf.compat.v1.random_normal([1024,5*10]))
    b_out = tf.Variable(b_alpha * tf.compat.v1.random_normal([5*10]))
    out=tf.add(tf.matmul(dense,w_out),b_out)

    return out

# 定义网络相关变量
X = tf.compat.v1.placeholder(tf.float32, [None,IMAGE_HEIGHT,IMAGE_WIDTH,1])
keep_prob = tf.compat.v1.placeholder(tf.float32)

# 创建模型
output = crack_captcha_cnn_network()

# 创建一个Saver对象来恢复模型
saver = tf.compat.v1.train.Saver()

# 创建一个TensorFlow会话
with tf.compat.v1.Session() as sess:
    # 加载模型的检查点
    saver.restore(sess,
    'zhu/Water_Num_recognize_20230905_2046.h5')

    # 在会话中使用模型进行推理
    # 假设你有一张待预测的图像，将其读取并预处理成适合模型的输入
    image_path = 'zhu/00045 (5).jpg'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    image = image.astype(np.float32)
    image = np.multiply(image, 1.0 / 255.0)
    image = np.array(image)
    image = np.expand_dims(image, -1)

    # 对图像进行推理
    result = sess.run(output, feed_dict={X: [image], keep_prob: 1.0})


    # 解码结果，将one-hot编码转化为标签文本
    def decode_result(result):
        vec1 = np.argmax(result, axis=1)
        text1 = ''
        for i in vec1:
            text1 += dictionary1[i]
        return text1


    decoded_text = decode_result(result)
    print("Predicted text:", decoded_text)