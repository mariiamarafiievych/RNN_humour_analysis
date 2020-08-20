import numpy as np
from numpy.random import randn
from data import train_data, test_data
import random

vocab=list(set([w for text in train_data.keys() for w in text.split(' ')]))
vocab_size=len(vocab)
print('%d unique words found'%vocab_size)

# Назначить индекс каждому слову
word_to_idx = { w: i for i, w in enumerate(vocab) }
idx_to_word = { i: w for i, w in enumerate(vocab) }
#print(word_to_idx['good']) # 16 (это может измениться)
#print(idx_to_word[0]) # грустно (это может измениться)


def createInputs(text):
    '''
    Возвращает массив унитарных векторов
    которые представляют слова в введенной строке текста
    - текст является строкой string
    - унитарный вектор имеет форму (vocab_size, 1)
    '''

    inputs = []
    for w in text.split(' '):
        v = np.zeros((vocab_size, 1))
        v[word_to_idx[w]] = 1
        inputs.append(v)

    return inputs

class RNN:
    # Классическая рекуррентная нейронная сеть

    def __init__(self, input_size, output_size, hidden_size=64):
        # Вес
        self.Whh = randn(hidden_size, hidden_size) / 1000
        self.Wxh = randn(hidden_size, input_size) / 1000
        self.Why = randn(output_size, hidden_size) / 1000

        # Смещения
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        '''
        Выполнение передачи нейронной сети при помощи входных данных
        Возвращение результатов вывода и скрытого состояния
        Вывод - это массив одного унитарного вектора с формой (input_size, 1)
        '''
        h = np.zeros((self.Whh.shape[0], 1))

        # Выполнение каждого шага в нейронной сети RNN
        for i, x in enumerate(inputs):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)

        # Compute the output
        y = self.Why @ h + self.by

        return y, h

    def forward(self, inputs):
        '''
        Выполнение фазы прямого распространения нейронной сети с
        использованием введенных данных.
        Возврат итоговой выдачи и скрытого состояния.
        - Входные данные в массиве однозначного вектора с формой (input_size, 1).
        '''
        h = np.zeros((self.Whh.shape[0], 1))

        self.last_inputs = inputs
        self.last_hs = {0: h}

        # Выполнение каждого шага нейронной сети RNN
        for i, x in enumerate(inputs):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            self.last_hs[i + 1] = h

        # Подсчет вывода
        y = self.Why @ h + self.by

        return y, h

    def backprop(self, d_y, learn_rate=2e-2):
        '''
        Выполнение фазы обратного распространения нейронной сети RNN.
        - d_y (dL/dy) имеет форму (output_size, 1).
        - learn_rate является вещественным числом float.
        '''
        n = len(self.last_inputs)

        # Вычисление dL/dWhy и dL/dby.
        d_Why = d_y @ self.last_hs[n].T
        d_by = d_y

        # Инициализация dL/dWhh, dL/dWxh, и dL/dbh к нулю.
        d_Whh = np.zeros(self.Whh.shape)
        d_Wxh = np.zeros(self.Wxh.shape)
        d_bh = np.zeros(self.bh.shape)

        # Вычисление dL/dh для последнего h.
        d_h = self.Why.T @ d_y

        # Обратное распространение во времени.
        for t in reversed(range(n)):
            # Среднее значение: dL/dh * (1 - h^2)
            temp = ((1 - self.last_hs[t + 1] ** 2) * d_h)

            # dL/db = dL/dh * (1 - h^2)
            d_bh += temp

            # dL/dWhh = dL/dh * (1 - h^2) * h_{t-1}
            d_Whh += temp @ self.last_hs[t].T

            # dL/dWxh = dL/dh * (1 - h^2) * x
            d_Wxh += temp @ self.last_inputs[t].T

            # Далее dL/dh = dL/dh * (1 - h^2) * Whh
            d_h = self.Whh @ temp

        # Отсекаем, чтобы предотвратить разрыв градиентов.
        for d in [d_Wxh, d_Whh, d_Why, d_bh, d_by]:
            np.clip(d, -1, 1, out=d)

        # Обновляем вес и смещение с использованием градиентного спуска.
        self.Whh -= learn_rate * d_Whh
        self.Wxh -= learn_rate * d_Wxh
        self.Why -= learn_rate * d_Why
        self.bh -= learn_rate * d_bh
        self.by -= learn_rate * d_by


def softmax(xs):
    # Применение функции Softmax для входного массива
    return np.exp(xs) / sum(np.exp(xs))


# Инициализация нашей рекуррентной нейронной сети RNN
rnn = RNN(vocab_size, 2)

inputs = createInputs('i am very good')
out, h = rnn.forward(inputs)
probs = softmax(out)
print(probs)  # [[0.50000095], [0.49999905]]

def sigmoid(x):
    # Наша функция активации: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))


class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        # Вводные данные о весе, добавление смещения
        # и последующее использование функции активации

        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)


weights = np.array([0, 1])  # w1 = 0, w2 = 1
bias = 4  # b = 4
n = Neuron(weights, bias)


class OurNeuralNetwork:
    """
    Нейронная сеть, у которой:
        - 2 входа
        - 1 скрытый слой с двумя нейронами (h1, h2)
        - слой вывода с одним нейроном (o1)
    У каждого нейрона одинаковые вес и смещение:
        - w = [0, 1]
        - b = 0
    """

    def __init__(self):
        weights = np.array([0, 1])
        bias = 0

        # Класс Neuron из предыдущего раздела
        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)

        # Вводы для о1 являются выводами h1 и h2
        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))

        return out_o1


network = OurNeuralNetwork()
x = np.array([2, 3])
print(network.feedforward(x))  # 0.7216325609518421


def processData(data, backprop=True):
    '''
    Возврат потери рекуррентной нейронной сети и точности для данных
    - данные представлены как словарь, что отображает текст как True или False.
    - backprop определяет, нужно ли использовать обратное распределение
    '''
    items = list(data.items())
    random.shuffle(items)

    loss = 0
    num_correct = 0

    for x, y in items:
        inputs = createInputs(x)
        target = int(y)

        # Прямое распределение
        out, _ = rnn.forward(inputs)
        probs = softmax(out)

        # Вычисление потери / точности
        loss -= np.log(probs[target])
        num_correct += int(np.argmax(probs) == target)

        if backprop:
            # Создание dL/dy
            d_L_d_y = probs
            d_L_d_y[target] -= 1

            # Обратное распределение
            rnn.backprop(d_L_d_y)

    return loss / len(data), num_correct / len(data)


# Цикл тренировки
for epoch in range(1000):
    train_loss, train_acc = processData(train_data)

    if epoch % 100 == 99:
        print('--- Epoch %d' % (epoch + 1))
        print('Train:\tLoss %.3f | Accuracy: %.3f' % (train_loss, train_acc))

        test_loss, test_acc = processData(test_data, backprop=False)
        print('Test:\tLoss %.3f | Accuracy: %.3f' % (test_loss, test_acc))
