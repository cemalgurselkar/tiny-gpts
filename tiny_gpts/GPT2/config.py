class GPTConfig:
    def __init__(self, **kwargs):
        self.block_size = kwargs.get('block_size', 128)
        self.n_layer = kwargs.get('n_layer', 4)
        self.n_head = kwargs.get('n_head', 4)
        self.n_embd = kwargs.get('n_embd', 256)
        self.dropout = kwargs.get('dropout', 0.1)
        self.bias = kwargs.get('bias', True)
        self.vocab_size = kwargs.get('vocab_size', 50257)

        self.batch_size = kwargs.get('batch_size', 4)
        self.learning_rate = kwargs.get('learning_rate', 3e-4)
        self.epochs = kwargs.get('epochs', 3)

        self.max_new_tokens = kwargs.get('max_new_tokens', 100)
        self.temperature = kwargs.get('temperature', 0.8)
        self.top_k = kwargs.get('top_k', 50)