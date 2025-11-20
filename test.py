import numpy as np

from utils import seeded_shuffle
from utils import seeded_unshuffle

s = [1, 2, 3, 4, 5]
c = [2, 1, 4, 1, 3]
seeded_shuffle(s, c)
print(s)

seeded_unshuffle(s, c)
print(s)



