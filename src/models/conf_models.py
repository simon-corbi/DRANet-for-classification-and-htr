from enum import Enum


class Activationfunction(Enum):
    SIGMOID = 1
    RELU = 2
    RELU_MAX_1 = 3
    NONE = 4
    TANH = 5
    MIN_MAX = 6
    RELU_SIGMOID = 7
    LEAKY_RELU = 8
    SILU = 9

    def __str__(self):
        return self.name
