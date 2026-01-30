import torch.nn as nn

from src.models.conf_models import Activationfunction


class CTCHead(nn.Module):
    def __init__(self, input_size, rnn_cfg, nclasses, dropout_last_fc_crnn=0.2, dropout_lstm=0.2):
        super(CTCHead, self).__init__()

        hidden, num_layers = rnn_cfg

        self.rec = nn.LSTM(input_size, hidden, num_layers=num_layers, bidirectional=True, dropout=dropout_lstm)
        self.fnl = nn.Sequential(nn.Dropout(dropout_last_fc_crnn), nn.Linear(2 * hidden, nclasses))

        self.cnn = nn.Conv2d(input_size, nclasses, kernel_size=(1, 3), stride=1, padding=(0, 1))

        self.dim_input_shortcut = input_size
        self.dim_input_main = 2 * hidden

    def forward(self, x):
        # x dimension: batch size, 256, 1,  nb frames
        y = x.permute(2, 3, 0, 1)[0]  # Output dimension: nb frames, batch size, 256
        y = self.rec(y)[0]
        after_blstm = y

        y = self.fnl(y)  # Output dimension: nb frames, batch size, alphabet size

        y_aux = self.cnn(x).permute(2, 3, 0, 1)[0]

        return [y, y_aux], after_blstm
