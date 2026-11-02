from collections import Counter
from random import Random
from scripts.bow import clean_update


class Perceptron:
    def __init__(self, r, bias, max_epochs, seed):
        self.r = r
        self.bias = bias
        self.max_epochs = max_epochs
        self.seed = seed

        # word -> weight, a word with no weight yet counts as 0
        self.weights = {}
        self.epoch = 0
        # number of mistakes in each epoch
        self.mistakes = []

    def to_bag(self, paragraph):
        bag = Counter()
        clean_update(bag, paragraph)
        return bag

    def sgn(self, v):
        if v > 0:
            return 1
        return -1

    def compute_v(self, x):
        # v = sum(wi * xi) + b(1)
        v = self.bias * 1
        for word in x:
            v += self.weights.get(word, 0) * x[word]
        return v

    def update_weights(self, x, d, y):
        # w(n+1) = w(n) + r(d - y)x
        for word in x:
            new_w = self.weights.get(word, 0) + self.r * (d - y) * x[word]
            self.weights[word] = new_w
        new_b = self.bias + self.r * (d - y) * 1
        self.bias = new_b

    def train(self, paragraphs, labels):
        '''
        labels
        0 : A
        1 : B
        '''
        inputs = []
        desired = []
        for i in range(len(paragraphs)):
            inputs.append(self.to_bag(paragraphs[i]))
            # the perceptron outputs -1 or 1
            if labels[i] == 1:
                desired.append(1)
            else:
                desired.append(-1)

        order = list(range(len(inputs)))
        random = Random(self.seed)

        # stops at max_epochs since the data might not be linearly separable
        while self.epoch < self.max_epochs:
            self.epoch += 1
            mistakes = 0

            # shuffled so the classes are not seen one after the other
            random.shuffle(order)
            for n in order:
                x = inputs[n]
                d = desired[n]
                y = self.sgn(self.compute_v(x))
                if y != d:
                    self.update_weights(x, d, y)
                    mistakes += 1

            self.mistakes.append(mistakes)
            print(f"Epoch {self.epoch}: {mistakes} mistakes")

            if mistakes == 0:
                break

    def classify_paragraph(self, p):
        '''
        returns
        0 : A
        1 : B
        '''
        if self.sgn(self.compute_v(self.to_bag(p))) == 1:
            return 1
        return 0
