import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='OOV')
text_corpus = ['bob ate apples, and pears', 'fred ate apples!']
tokenizer.fit_on_texts(text_corpus)
new_texts = ['bob ate pears', 'fred ate pears', 'bob ate bacon']
print(tokenizer.texts_to_sequences(new_texts))
print(tokenizer.word_index)

# Variable initializer
print(tf.compat.v1.get_variable('v1', shape=(1, 3)))
print(tf.compat.v1.get_variable('v2', shape=(2,), dtype=tf.int64))

# Variable initializer using init
init1 = tf.random.uniform((5, 10), minval=-1, maxval=2)
init2 = tf.zeros([5, 4])
init3 = tf.zeros([7])
v = tf.compat.v1.get_variable('v1', initializer=init2)

# embedding_lookup
emb_mat = tf.compat.v1.get_variable('v1', shape=(5, 10))
word_ids = tf.constant([0, 3])
emb_vecs = tf.nn.embedding_lookup(emb_mat, word_ids)
print(emb_vecs)

params1 = tf.constant([1, 2])
params2 = tf.constant([10, 20])
ids = tf.constant([2, 0, 2, 1, 2, 3])
result = tf.nn.embedding_lookup([params1, params2], ids)


# Skip-gram embedding model
class EmbeddingModel(object):
    # Model Initialization
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.vocab_size)

    # Forward run of the embedding model to retrieve embeddings
    def forward(self, target_ids):
        initial_bounds = 0.5 / self.embedding_dim
        initializer = tf.random.uniform(
            [self.vocab_size, self.embedding_dim],
            minval=-initial_bounds,
            maxval=initial_bounds)
        self.embedding_matrix = tf.compat.v1.get_variable('embedding_matrix', initializer=initializer)
        embeddings = tf.compat.v1.nn.embedding_lookup(self.embedding_matrix, target_ids)
        return embeddings

    # Compute cosine similarites between the word's embedding
    # and all other embeddings for each vocabulary word
    def compute_cos_sims(self, word, training_texts):
        self.tokenizer.fit_on_texts(training_texts)
        word_id = self.tokenizer.word_index[word]
        word_embedding = self.forward([word_id])
        normalized_embedding = tf.math.l2_normalize(word_embedding)
        normalized_matrix = tf.math.l2_normalize(self.embedding_matrix, axis=1)
        cos_sims = tf.linalg.matmul(normalized_embedding, normalized_matrix, transpose_b=True)
        return cos_sims

    # Compute K-nearest neighbors for input word
    def k_nearest_neighbors(self, word, k, training_texts):
        neighbors = self.compute_cos_sims(word, training_texts)
        neighbors.sort()
        return neighbors[:k]
