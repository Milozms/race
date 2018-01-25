import numpy as np
with open('../npsave/epoch0batch0att_weight_', 'rb') as f:
    w = np.load(f)
    print(w)
    print(np.sum(w))
    print(np.sum(w, 2))
    print(np.sum(w, (1, 2)))