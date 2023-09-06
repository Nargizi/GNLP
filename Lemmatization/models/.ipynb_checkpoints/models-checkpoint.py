from random import random
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, vocab_dim: int, emb_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()

        # set dimensions
        self.hidden_size = hidden_dim
        self.embedding_size = emb_dim
        self.vocab_size = vocab_dim
        self.num_layers = num_layers

        # initialize layers
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.gru = nn.GRU(self.embedding_size, self.hidden_size, num_layers=self.num_layers, batch_first=True)

    def forward(self, input: torch.Tensor, hidden: Optional[torch.Tensor] = None):
        embedded = self.embedding(input)
        if hidden is not None:
            output, hidden = self.gru(embedded, hidden)
        else:
            output, hidden = self.gru(embedded)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int, vocab_dim: int, num_layers: int = 1):
        super(DecoderRNN, self).__init__()
        # set dimensions
        self.hidden_size = hidden_dim
        self.embedding_size = emb_dim
        self.output_size = vocab_dim
        self.num_layers = num_layers

        # initialize layers
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor):
        embedding = F.relu(self.embedding(input))
        output, hidden = self.gru(embedding, hidden)
        pred = self.softmax(self.out(output))
        return pred, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, sow_token: int,
                 max_length: int, device: str = 'cpu', teacher_forcing_ratio: float = 0.5):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        #         self.decoder.embedding = self.encoder.embedding  #
        self.sow_token = sow_token
        self.length = max_length
        self.device = device
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, source: torch.Tensor, target: Optional[torch.Tensor] = None):
        input_length = source.size(1)
        batch_size = source.size(0)
        vocab_size = self.encoder.vocab_size

        # initialize a variable to hold the predicted outputs
        outputs = []
        encoder_output, encoder_hidden = self.encoder(source[:, 0].unsqueeze(1))
        # encode every word in a sentence
        for i in range(1, input_length):
            encoder_output, encoder_hidden = self.encoder(source[:, i].unsqueeze(1), encoder_hidden)

        # use the encoder’s hidden layer as the decoder hidden
        decoder_hidden = encoder_hidden.to(self.device)

        # add a start_of_the_word token before the first predicted word
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=self.device).fill_(self.sow_token)

        # max_length: int = self.length
        for t in range(self.length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs.append(decoder_output)
            _, topi = decoder_output.topk(1)
            if target is not None and target.size(1) > t:
                teacher_force: bool = random() < self.teacher_forcing_ratio
                target_input = target[:, t].unsqueeze(1)
                decoder_input = (target_input if teacher_force else topi.squeeze(-1).detach())
            else:
                decoder_input = topi.squeeze(-1).detach()
        return torch.cat(outputs, dim=1)


    class BaseModel(nn.Module):
        pass



