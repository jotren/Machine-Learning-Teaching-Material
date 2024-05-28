# Transformers Encoder Maths

The transformer architecture is what has allowed the jump in large language models. It is very important to understand the maths behind semantic space and context finding. If the word appe cppears a lot in a sentnce with the word phone, these two word vectors need to move closer to eachother.

<img src="./images/Semantic Space.png" alt="alt text" width="400"/> 

This is how encoders manage to capture context and complex relationships between words.

In order to understand this, you'll need to have a knowledge of:

- Matrices
- Matrix Manipulations
- Identity Matrices
- Softmax Calculations

### Transformer Encoder Example with "phone," "apple," and "orange"

The key thing to understand about transformers is that the encoder is moving words through semantic space. The "attention mechanism" essentially moves words closer together when they appear more frequently in the same sentence or context. Let's take the example of phone, apple and orange, apple has two different meanings depending on the context, how can we train a machine to understand this.

1``

We represent each word as a vector in the "fruitness" and "phoneness" space. Let's assume:

$$
\text{"apple"} = \begin{bmatrix} 0 & 3 \end{bmatrix} \quad (\text{0 fruitness, 3 phoneness})
$$

$$
\text{"phone"} = \begin{bmatrix} 2 & 0.5 \end{bmatrix} \quad (\text{2 fruitness, 0.5 phoneness})
$$

$$
\text{"orange"} = \begin{bmatrix} 2 & 0 \end{bmatrix} \quad (\text{2 fruitness, 0 phoneness})
$$

Thus, our input matrix \(X\) (with each row corresponding to a word vector) is:

$$
X = \begin{bmatrix}
0 & 3 \\
2 & 0.5 \\
2 & 0
\end{bmatrix}
$$

### Calculation of \(Q\), \(K\), \(V\)

Assume the identity matrix as the weights for simplicity. This means:

$$
W_Q = W_K = W_V = I
$$

Then:

$$
Q = XW_Q = X
$$

$$
K = XW_K = X
$$

$$
V = XW_V = X
$$

So, \(Q\), \(K\), and \(V\) are:

$$
Q = \begin{bmatrix}
0 & 3 \\
2 & 0.5 \\
2 & 0
\end{bmatrix}
$$

$$
K = \begin{bmatrix}
0 & 3 \\
2 & 0.5 \\
2 & 0
\end{bmatrix}
$$

$$
V = \begin{bmatrix}
0 & 3 \\
2 & 0.5 \\
2 & 0
\end{bmatrix}
$$

### Scaled Dot-Product Attention

The attention score for each word pair is computed as:

$$
\text{Attention}(Q, K) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)
$$

where \(d_k\) is the dimension of the key vectors, which is 2 in this case.

First, we compute the dot product \(QK^T\):

$$
QK^T = \begin{bmatrix}
0 & 3 \\
2 & 0.5 \\
2 & 0
\end{bmatrix}
\begin{bmatrix}
0 & 3 & 2 \\
3 & 0.5 & 0
\end{bmatrix} = \begin{bmatrix}
9 & 1.5 & 0 \\
1.5 & 4.25 & 4 \\
0 & 4 & 4
\end{bmatrix}
$$

Next, we scale by \(\sqrt{d_k}\):

$$
\frac{QK^T}{\sqrt{d_k}} = \frac{1}{\sqrt{2}} \begin{bmatrix}
9 & 1.5 & 0 \\
1.5 & 4.25 & 4 \\
0 & 4 & 4
\end{bmatrix} = \begin{bmatrix}
\frac{9}{\sqrt{2}} & \frac{1.5}{\sqrt{2}} & 0 \\
\frac{1.5}{\sqrt{2}} & \frac{4.25}{\sqrt{2}} & \frac{4}{\sqrt{2}} \\
0 & \frac{4}{\sqrt{2}} & \frac{4}{\sqrt{2}}
\end{bmatrix}
$$

Applying softmax to each row to get the attention weights:

$$
\text{softmax}\left(\begin{bmatrix}
\frac{9}{\sqrt{2}} & \frac{1.5}{\sqrt{2}} & 0 \\
\frac{1.5}{\sqrt{2}} & \frac{4.25}{\sqrt{2}} & \frac{4}{\sqrt{2}} \\
0 & \frac{4}{\sqrt{2}} & \frac{4}{\sqrt{2}}
\end{bmatrix}\right)
$$

### Attention Weight Calculation (Simplified)

We calculate the softmax for each row separately:

For the first row:

$$
\text{softmax}\left(\frac{9}{\sqrt{2}}, \frac{1.5}{\sqrt{2}}, 0\right)
$$

$$
\text{softmax}\left(\frac{9}{\sqrt{2}}, \frac{1.5}{\sqrt{2}}, 0\right) = \left(\frac{e^{\frac{9}{\sqrt{2}}}}{e^{\frac{9}{\sqrt{2}}} + e^{\frac{1.5}{\sqrt{2}}} + 1}, \frac{e^{\frac{1.5}{\sqrt{2}}}}{e^{\frac{9}{\sqrt{2}}} + e^{\frac{1.5}{\sqrt{2}}} + 1}, \frac{1}{e^{\frac{9}{\sqrt{2}}} + e^{\frac{1.5}{\sqrt{2}}} + 1}\right)
$$

Calculating the values:

$$
e^{\frac{9}{\sqrt{2}}} \approx 577.64, \quad e^{\frac{1.5}{\sqrt{2}}} \approx 2.89
$$

$$
\left(\frac{577.64}{577.64 + 2.89 + 1}, \frac{2.89}{577.64 + 2.89 + 1}, \frac{1}{577.64 + 2.89 + 1}\right) = \left(0.99, 0.01, 0.00\right)
$$

For the second row:

$$
\text{softmax}\left(\frac{1.5}{\sqrt{2}}, \frac{4.25}{\sqrt{2}}, \frac{4}{\sqrt{2}}\right)
$$

$$
\text{softmax}\left(\frac{1.5}{\sqrt{2}}, \frac{4.25}{\sqrt{2}}, \frac{4}{\sqrt{2}}\right) = \left(\frac{e^{\frac{1.5}{\sqrt{2}}}}{e^{\frac{1.5}{\sqrt{2}}} + e^{\frac{4.25}{\sqrt{2}}} + e^{\frac{4}{\sqrt{2}}}}, \frac{e^{\frac{4.25}{\sqrt{2}}}}{e^{\frac{1.5}{\sqrt{2}}} + e^{\frac{4.25}{\sqrt{2}}} + e^{\frac{4}{\sqrt{2}}}}, \frac{e^{\frac{4}{\sqrt{2}}}}{e^{\frac{1.5}{\sqrt{2}}} + e^{\frac{4.25}{\sqrt{2}}} + e^{\frac{4}{\sqrt{2}}}}\right)
$$

Calculating the values:

$$
e^{\frac{1.5}{\sqrt{2}}} \approx 2.89, \quad e^{\frac{4.25}{\sqrt{2}}} \approx 20.09, \quad e^{\frac{4}{\sqrt{2}}} \approx 16.99
$$

$$
\left(\frac{2.89}{2.89 + 20.09 + 16.99}, \frac{20.09}{2.89 + 20.09 + 16.99}, \frac{16.99}{2.89 + 20.09 + 16.99}\right) = \left(0.07, 0.50, 0.42\right)
$$

For the third row:

$$
\text{softmax}\left(0, \frac{4}{\sqrt{2}}, \frac{4}{\sqrt{2}}\right)
$$

$$
\text{softmax}\left(0, \frac{4}{\sqrt{2}}, \frac{4}{\sqrt{2}}\right) = \left(\frac{1}{1 + e^{\frac{4}{\sqrt{2}}} + e^{\frac{4}{\sqrt{2}}}}, \frac{e^{\frac{4}{\sqrt{2}}}}{1 + e^{\frac{4}{\sqrt{2}}} + e^{\frac{4}{\sqrt{2}}}}, \frac{e^{\frac{4}{\sqrt{2}}}}{1 + e^{\frac{4}{\sqrt{2}}} + e^{\frac{4}{\sqrt{2}}}}\right)
$$

Calculating the values:

$$
e^{\frac{4}{\sqrt{2}}} \approx 16.99
$$

$$
\left(\frac{1}{1 + 16.99 + 16.99}, \frac{16.99}{1 + 16.99 + 16.99}, \frac{16.99}{1 + 16.99 + 16.99}\right) = \left(0.03, 0.49, 0.49\right)
$$

Result being:

$$
\text{Attention}(Q, K) = \begin{bmatrix}
0.99 & 0.01 & 0.00 \\
0.07 & 0.50 & 0.42 \\
0.03 & 0.49 & 0.49
\end{bmatrix}
$$

### Compute the Output

The output for each word is computed as:

$$
\text{Output} = \text{Attention}(Q, K)V
$$

Given the attention matrix and the value matrix \( V \):

$$
V = \begin{bmatrix}
0 & 3 \\
2 & 0.5 \\
2 & 0
\end{bmatrix}
$$

Multiplying the attention matrix with \( V \):

$$
\text{Output} = \begin{bmatrix}
0.99 & 0.01 & 0.00 \\
0.07 & 0.50 & 0.42 \\
0.03 & 0.49 & 0.49
\end{bmatrix}
\begin{bmatrix}
0 & 3 \\
2 & 0.5 \\
2 & 0
\end{bmatrix}
$$

Calculating each element:

For the first row:

$$
\begin{bmatrix}
0.99 \cdot 0 + 0.01 \cdot 2 + 0.00 \cdot 2 & 0.99 \cdot 3 + 0.01 \cdot 0.5 + 0.00 \cdot 0
\end{bmatrix}
= \begin{bmatrix}
0.02 & 2.98
\end{bmatrix}
$$

For the second row:

$$
\begin{bmatrix}
0.07 \cdot 0 + 0.50 \cdot 2 + 0.42 \cdot 2 & 0.07 \cdot 3 + 0.50 \cdot 0.5 + 0.42 \cdot 0
\end{bmatrix}
= \begin{bmatrix}
1.84 & 0.44
\end{bmatrix}
$$

For the third row:

$$
\begin{bmatrix}
0.03 \cdot 0 + 0.49 \cdot 2 + 0.49 \cdot 2 & 0.03 \cdot 3 + 0.49 \cdot 0.5 + 0.49 \cdot 0
\end{bmatrix}
= \begin{bmatrix}
1.96 & 0.26
\end{bmatrix}
$$

So the final output matrix is:

$$
\text{Output} = \begin{bmatrix}
0.02 & 2.98 \\
1.84 & 0.44 \\
1.96 & 0.26
\end{bmatrix}
$$

### Example: Moving Apple and Phone Closer

We want to see how "apple" and "phone" move closer when they appear in similar sentences.

Consider sentences where "apple" and "phone" appear together frequently. The attention mechanism will cause the vectors of "apple" and "phone" to influence each other more strongly.

As the transformer trains on such sentences, it learns to adjust the weights such that the representations of "apple" and "phone" become more similar in the "fruitness" and "phoneness" space.

Initially:

$$
\text{"phone"} = \begin{bmatrix} 2 & 0.5 \end{bmatrix}
$$

$$
\text{"apple"} = \begin{bmatrix} 0 & 3 \end{bmatrix}
$$

After training on similar contexts, their vectors might move closer. For example:

$$
\text{"phone"} = \begin{bmatrix} 1.84 & 0.44 \end{bmatrix}
$$

$$
\text{"apple"} = \begin{bmatrix} 0.02 & 2.98 \end{bmatrix}
$$

This change would be a result of the transformer's learned weights adjusting to better represent the contextual similarity between "apple" and "phone."

Here are the final outputs calculated from the attention mechanism:

$$
\text{Output} = \begin{bmatrix}
0.02 & 2.98 \\
1.84 & 0.44 \\
1.96 & 0.26
\end{bmatrix}
$$
