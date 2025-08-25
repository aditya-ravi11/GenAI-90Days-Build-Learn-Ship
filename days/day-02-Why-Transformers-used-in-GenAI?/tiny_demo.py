# Tiny sanity check: scaled dot-product attention on small tensors
import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    dk = Q.shape[-1]
    scores = (Q @ K.T) / np.sqrt(dk)
    weights = softmax(scores, axis=-1)
    out = weights @ V
    return out, weights

def main():
    # Toy example: 2 tokens, d_k = d_v = 4
    Q = np.array([[1.,0.,1.,0.],
                  [0.,1.,0.,1.]])
    K = np.array([[1.,0.,1.,0.],
                  [1.,1.,0.,0.]])
    V = np.array([[1.,2.,0.,0.],
                  [0.,1.,3.,0.]])

    out, w = scaled_dot_product_attention(Q, K, V)
    print("Weights:\n", np.round(w, 3))
    print("Output:\n", np.round(out, 3))

if __name__ == "__main__":
    main()
