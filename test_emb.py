
import numpy as np
emb = np.random.randn(5000, 128)
idx = np.random.randint(0, 5000, (2, 63))
x = emb[idx]
print(f"Shape: {x.shape}")
