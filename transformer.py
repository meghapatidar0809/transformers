import math
import random



###################################  Function to perform matrix multiplication ################################## 

def matrix_mult(A, B):
    rows_A, cols_A = len(A), len(A[0])   # Get dimensions of matrix A
    rows_B, cols_B = len(B), len(B[0])   # Get dimensions of matrix B
    
    # Ensuring that matrices can be multiplied (cols of A must match rows of B)
    assert cols_A == rows_B, "Matrix dimensions do not match for multiplication!"
    
    # Initializing result matrix with all zeros
    result = [[0.0 for _ in range(cols_B)] for _ in range(rows_A)]
    
    # Matrix multiplication
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
                
    return result

 
 

###################################  Transpose of matrix ################################## 

def matrix_transpose(A):
    # Get the number of rows of input matrix
    num_rows = len(A)
    
    # Get the number of columns of input matrix
    num_cols = len(A[0])
    
    # Creating a new matrix with swapped dimensions (columns become rows)
    transpose = []
    
    # Loop through each column index of the original matrix
    for i in range(num_cols):
        # Creating a new row for the transposed matrix
        new_row = []
        
        # Loop through each row index of the original matrix and append the corresponding element to the new row
        for j in range(num_rows):
            new_row.append(A[j][i])   
        
        transpose.append(new_row)     # Append the new row to the transposed matrix
    
    return transpose # Return the transposed matrix




###################################  Softmax activation function ################################## 
def softmax(vector):
    # Softmax converts a vector of real numbers into a probability distribution.

    # Compute the exponentials of each element
    exp_values = [math.exp(val) for val in vector]
    
    # Compute the sum of exponentials
    sum_exp = sum(exp_values)
    
    #  Normalization
    return [val / sum_exp for val in exp_values]




################################### Layer Normalization ################################### 
def normalize_layer(values, eps=1e-5):
    # Normalization helps stabilize learning by maintaining zero mean and unit variance.

    # Compute the mean of the values
    mean = sum(values) / len(values)
    
    # Compute the variance of the values
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    
    #  Normalization
    return [(x - mean) / math.sqrt(variance + eps) for x in values]




####################################  Fully Connected Layer ################################### 
class LinearLayer:
    # Applying a linear transformation: output = input * weights + bias
    
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim      # Number of input neurons
        self.output_dim = output_dim    # Number of output neurons
        
        # Initialize weights randomly between -0.1 and 0.1
        self.weights = [[random.uniform(-0.1, 0.1) for _ in range(output_dim)] 
                        for _ in range(input_dim)]
        
        # Initialize biases to zero
        self.biases = [0.0 for _ in range(output_dim)]
    
    def forward(self, x):
        # Computes the forward pass of the linear layer by computing the weighted sum of inputs for each output neuron
        return [
            sum(x[i] * self.weights[i][j] for i in range(self.input_dim)) + self.biases[j]
            for j in range(self.output_dim)
        ]



####################################  Positional Encoding Mechanism ################################### 
class PositionEncoding:
    # Implements sinusoidal positional encoding for a Transformer model.

    def __init__(self, model_dim, max_length=5000):
        self.model_dim = model_dim

        # Initialize an empty encoding matrix of shape (max_length, model_dim)
        self.encoding_matrix = [[0.0 for _ in range(model_dim)] for _ in range(max_length)]
        
        # Compute positional encodings for each position
        for pos in range(max_length):
            for i in range(0, model_dim, 2):  # Process even indices separately
                self.encoding_matrix[pos][i] = math.sin(pos / (10000 ** (i / model_dim))) # Compute sine for even indices
                
                # Compute cosine for odd indices (if within range)
                if i + 1 < model_dim:
                    self.encoding_matrix[pos][i+1] = math.cos(pos / (10000 ** (i / model_dim)))
    
    def forward(self, tokens):
        # Applies positional encoding to the input token embeddings.
        return [
            [tokens[i][j] + self.encoding_matrix[i][j] for j in range(self.model_dim)]
            for i in range(len(tokens))
        ]




####################################  Multi-Head Attention Module ################################### 
class MultiHeadAttention:
    #Implements a Multi-Head Self-Attention mechanism.

    def __init__(self, model_dim=256, num_heads=8):
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads  # Ensure even split of dimensions

        #  Define linear layers for query, key, and value projections
        self.query_layer = LinearLayer(model_dim, model_dim)
        self.key_layer = LinearLayer(model_dim, model_dim)
        self.value_layer = LinearLayer(model_dim, model_dim)

        # Define the output projection layer
        self.output_layer = LinearLayer(model_dim, model_dim)
    
    def split_heads(self, matrix):
        # Splits the input matrix into multiple heads for multi-head attention.
        return [
            [matrix[i][j:j + self.head_dim] for i in range(len(matrix))]  
            for j in range(0, self.model_dim, self.head_dim)
        ]
    
    def merge_heads(self, heads):
        # Merges multiple attention heads back into a single representation.
        return [
            [val for head in heads for val in head[i]]  
            for i in range(len(heads[0]))
        ]
    
    def forward(self, query, key, value):
        # Computes the multi-head attention mechanism.
        
        # Compute Query (Q), Key (K), and Value (V) matrices
        Q = [self.query_layer.forward(q) for q in query]
        K = [self.key_layer.forward(k) for k in key]
        V = [self.value_layer.forward(v) for v in value]

        # Split Q, K, and V matrices into multiple attention heads
        heads_Q = self.split_heads(Q)
        heads_K = self.split_heads(K)
        heads_V = self.split_heads(V)
        
        output_heads = []  # Store attention outputs for each head

        # Compute scaled dot-product attention for each head
        for h in range(self.num_heads):
            head_output = []
            for i, q in enumerate(heads_Q[h]):
                # Compute attention scores (dot product of Q and K)
                scores = [
                    sum(q[j] * k[j] for j in range(self.head_dim)) / math.sqrt(self.head_dim)
                    for k in heads_K[h]
                ]
                
                # Apply softmax to get attention weights
                attn_weights = softmax(scores)
                
                # Compute weighted sum of value vectors
                weighted_sum = [
                    sum(attn_weights[j] * heads_V[h][j][k] for j in range(len(heads_V[h])))
                    for k in range(self.head_dim)
                ]
                
                head_output.append(weighted_sum)
            
            output_heads.append(head_output)
        
        # Merge attention heads back together
        combined_heads = self.merge_heads(output_heads)

        # Apply the final linear projection
        return [self.output_layer.forward(token) for token in combined_heads]




####################################  Transformer Model ################################### 
class Transformer:
    #Implements a basic Transformer model with an encoder-decoder architecture.
    
    def __init__(self, vocab_src, vocab_tgt, model_dim=256, num_heads=8):
        self.model_dim = model_dim
        
        # Initialize source and target token embeddings randomly
        self.src_embedding = {
            i: [random.uniform(-0.1, 0.1) for _ in range(model_dim)]
            for i in range(vocab_src)
        }
        self.tgt_embedding = {
            i: [random.uniform(-0.1, 0.1) for _ in range(model_dim)]
            for i in range(vocab_tgt)
        }
        
        #bInitialize positional encoding for sequence processing
        self.position_encoder = PositionEncoding(model_dim)
        
        # Define the encoder and decoder using Multi-Head Attention
        self.encoder = MultiHeadAttention(model_dim, num_heads)
        self.decoder = MultiHeadAttention(model_dim, num_heads)
        
        # Define the final output projection layer
        self.final_layer = LinearLayer(model_dim, vocab_tgt)
    
    def forward(self, src_tokens, tgt_tokens):
        #Performs a forward pass of the Transformer model.
        
        # Convert source and target token indices to embeddings
        src_embeddings = self.position_encoder.forward(
            [self.src_embedding[t] for t in src_tokens]
        )
        tgt_embeddings = self.position_encoder.forward(
            [self.tgt_embedding[t] for t in tgt_tokens]
        )
        
        # Pass source embeddings through the encoder
        encoder_out = self.encoder.forward(src_embeddings, src_embeddings, src_embeddings)
        
        # Pass target embeddings through the decoder, attending to encoder output
        decoder_out = self.decoder.forward(tgt_embeddings, encoder_out, encoder_out)
        
        # Apply the final linear layer to transform output to vocabulary space
        return [self.final_layer.forward(token) for token in decoder_out]




####################################  Main Function ################################### 
if __name__ == "__main__":
    # Initializes a Transformer and processes a random source-target token sequence.
    
    # Define vocabulary sizes for source and target languages
    vocab_size_source = 10000
    vocab_size_target = 10000
    
    # Define sequence lengths for source and target sentences
    sequence_length_source = 10
    sequence_length_target = 8
    
    # Create a Transformer model instance
    model = Transformer(vocab_size_source, vocab_size_target)
    
    # Generate random token sequences for testing
    src_sample = [random.randint(0, vocab_size_source - 1) for _ in range(sequence_length_source)]
    tgt_sample = [random.randint(0, vocab_size_target - 1) for _ in range(sequence_length_target)]
    
    # Perform a forward pass through the model
    output = model.forward(src_sample, tgt_sample)
    
    # Print the output shape to verify correctness
    print(f"Transformer output shape: ({sequence_length_target}, {vocab_size_target})")