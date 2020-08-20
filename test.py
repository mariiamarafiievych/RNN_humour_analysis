import random
import RNN
import numpy as np
import main
import softmax
import data


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
        inputs = main.createInputs(x)
        target = int(y)

        # Прямое распределение
        out, _ = RNN.forward(inputs)
        probs = softmax.softmax(out)

        # Вычисление потери / точности
        loss -= np.log(probs[target])
        num_correct += int(np.argmax(probs) == target)

        if backprop:
            # Создание dL/dy
            d_L_d_y = probs
            d_L_d_y[target] -= 1

            # Обратное распределение
            RNN.backprop(d_L_d_y)

    return loss / len(data), num_correct / len(data)


# Цикл тренировки
for epoch in range(1000):
    train_loss, train_acc = processData(data.train_data)

    if epoch % 100 == 99:
        print('--- Epoch %d' % (epoch + 1))
        print('Train:\tLoss %.3f | Accuracy: %.3f' % (train_loss, train_acc))

        test_loss, test_acc = processData(data.test_data, backprop=False)
        print('Test:\tLoss %.3f | Accuracy: %.3f' % (test_loss, test_acc))