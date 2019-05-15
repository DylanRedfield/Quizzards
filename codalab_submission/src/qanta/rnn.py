# Standard library imports
# Third party imports
import torch
from torch import nn
# Local application imports



class RNNBuzzer(nn.Module):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    We use a LSTM for our buzzer.
    """

    #### You don't need to change the parameters for the model

    # n_input represents dimensionality of each specific feature vector, n_output is number of labels
    def __init__(self, n_input=10, n_hidden=50, n_output=2, dropout=0.5):
        super(RNNBuzzer, self).__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.n_output = n_output

        ### Your Code Here ---
        # define lstm layer, going from input to hidden. Remember to have batch_first=True.
        self.lstm = nn.LSTM(n_input, n_hidden, batch_first=True)

        # define linear layer going from hidden to output.
        self.hidden_to_label = nn.Linear(n_hidden, n_output)

        # self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # ---you can add other things like dropout between two layers, but do so in forward function below,
        # as we have to perform an extra step on the output of LSTM before going to linear layer(s).
        # The MODEL FOR TEST CASES is just single layer LSTM followed by 1 linear layer - do not add anything else for the purpose of passing test cases!!

    def forward(self, X):
        # get the batch size and sequence length (max length of the batch)
        # dim of X: batch_size x batch_max_len x input feature vec dim
        # batch_size, seq_len, _ = X.size()
        print('X.size()', X.size())

        ###Your code here --
        # Get the output of LSTM - (output dim: batch_size x batch_max_len x lstm_hidden_dim)

        lstm_out, lstm_hidden = self.lstm(X)

        # reshape (before passing to linear layer) so that each row contains one token
        # essentially, flatten the output of LSTM
        # dim will become batch_size*batch_max_len x lstm_hidden_dim

        lstm_out = lstm_out.contiguous().view(-1, self.n_hidden)

        # Get logits from the final linear layer
        logits = self.hidden_to_label(lstm_out)

        return logits