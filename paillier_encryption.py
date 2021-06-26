
from numpy import random
from sklearn.datasets import load_diabetes
import phe as paillier
import numpy as np
class Server:
    """Private key holder. Decrypts the average gradient"""

    def __init__(self, key_length):
         keypair = paillier.generate_paillier_keypair(n_length=key_length)
         self.pubkey, self.privkey = keypair

    def decrypt_aggregate(self, input_model, n_clients):
        return decrypt_vector(self.privkey, input_model) / n_clients
    
def encrypt_vector(public_key, x):
    rel=[]
    print(x)
    for i in x:
        rel.append(public_key.encrypt(i))

    return rel


def decrypt_vector(private_key, x):
    return np.array([private_key.decrypt(i) for i in x])


def sum_encrypted_vectors(x, y):
    if len(x) != len(y):
        raise ValueError('Encrypted vectors must have the same size')
    return [x[i] + y[i] for i in range(len(x))]

class Client:
    """Runs linear regression with local data or by gradient steps,
    where gradient can be passed in.
    Using public key can encrypt locally computed gradients.
    """

    def __init__(self, name, X, y, pubkey,sizes):
        self.name = name
        self.pubkey = pubkey
        X=[[1,0,1],
        [1,0,0],
        [0,1,0],
        [0,0,0],
        [0,1,0]]
        X=np.matrix(X)
        y=[[0,1,0],
        [0,1,1],
        [1,0,1],
        [1,1,1],
        [1,0,1]]
        y=np.matrix(y)
        self.weights = np.zeros(X.shape[1])
        """
        :param sizes: list类型，储存每层神经网络的神经元数目
                      譬如说：sizes = [2, 3, 2] 表示输入层有两个神经元、
                      隐藏层有3个神经元以及输出层有2个神经元
        """
        # 有几层神经网络
        self.num_layers = len(sizes)
        self.sizes = sizes
        # 除去输入层，随机产生每层中 y 个神经元的 biase 值（0 - 1）
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # 随机产生每条连接线的 weight 值（0 - 1）
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def fit(self, n_iter, eta=0.01):
        """Linear regression for n_iter"""
        for _ in range(n_iter):
            gradient = self.compute_gradient()
            self.gradient_step(gradient, eta)

    def gradient_step(self, gradient, eta=0.01):
        """Update the model with the given gradient"""
        self.weights -= eta * gradient

    def compute_gradient(self):
        """Compute the gradient of the current model using the training set
        """

        return [self.nabla_b,self.nabla_w]

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """
        随机梯度下降
        :param training_data: 输入的训练集
        :param epochs: 迭代次数
        :param mini_batch_size: 小样本数量
        :param eta: 学习率
        :param test_data: 测试数据集
        """
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            # 搅乱训练集，让其排序顺序发生变化
            random.shuffle(training_data)
            # 按照小样本数量划分训练集
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                # 根据每个小样本来更新 w 和 b，代码在下一段
                nabla_b,nabla_w=self.cal_bw_with_mini_batch(mini_batch, eta)
                #应该在这里上传梯度
                self.mod_grad(nabla_b,nabla_w,eta,mini_batch)
            # 输出测试每轮结束后，神经网络的准确度
            if test_data:
                print ("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print ("Epoch {0} complete".format(j))

    def cal_bw_with_mini_batch(self, mini_batch, eta):
        """
        更新 w 和 b 的值
        :param mini_batch: 一部分的样本
        :param eta: 学习率
        """
        # 根据 biases 和 weights 的行列数创建对应的全部元素值为 0 的空矩阵
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            # 根据样本中的每一个输入 x 的其输出 y，计算 w 和 b 的偏导数
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # 累加储存偏导值 delta_nabla_b 和 delta_nabla_w
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.nabla_b=nabla_b
        self.nabla_w=nabla_w
        return nabla_b,nabla_w

    def mod_grad(self,nabla_b,nabla_w,eta,mini_batch):
        # 更新根据累加的偏导值更新 w 和 b，这里因为用了小样本，
        # 所以 eta 要除于小样本的长度
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def feedforward(self, a):
        """
        前向传输计算每个神经元的值
        :param a: 输入值
        :return: 计算后每个神经元的值
        """
        for b, w in zip(self.biases, self.weights):
            # 加权求和以及加上 biase
            a = sigmoid(np.dot(w, a) + b)
        return a

    def evaluate(self, test_data):
        # 获得预测结果
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        # 返回正确识别的个数
        return sum(int(x == y) for (x, y) in test_results)
    def cost_derivative(self, output_activations, y):
        """
        二次损失函数
        :param output_activations:
        :param y:
        :return:
        """
        return (output_activations-y)

    def backprop(self, x, y):
        """
        :param x:
        :param y:
        :return:
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # 前向传输
        activation = x
        # 储存每层的神经元的值的矩阵，下面循环会 append 每层的神经元的值
        activations = [x]
        # 储存每个未经过 sigmoid 计算的神经元的值
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # 求 δ 的值
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        # 乘于前一层的输出值
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            # 从倒数第 **l** 层开始更新，**-l** 是 python 中特有的语法表示从倒数第 l 层开始计算
            # 下面这里利用 **l+1** 层的 δ 值来计算 **l** 的 δ 值
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)


    def predict(self, X):
        """Score test data"""
        #print(X)
        return X.dot(self.weights)

    def encrypted_gradient(self, sum_to=None):
        """Compute and encrypt gradient.
        When `sum_to` is given, sum the encrypted gradient to it, assumed
        to be another vector of the same size
        """
        gradient = self.compute_gradient()
        encrypted_gradient = encrypt_vector(self.pubkey, gradient)

        if sum_to is not None:
            return sum_encrypted_vectors(sum_to, encrypted_gradient)
        else:
            return encrypted_gradient
def sigmoid(z):
    """
    求 sigmoid 函数的值
    :param z:
    :return:
    """
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """
    求 sigmoid 函数的导数
    :param z:
    :return:
    """
    return sigmoid(z)*(1-sigmoid(z))