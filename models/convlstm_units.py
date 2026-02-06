import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedConvLSTMUnit(nn.Module):
    """Standard ConvLSTM implementation based on paper formulas"""

    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        self.W_xi = nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=padding)
        self.W_hi = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding)
        self.W_ci = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding)

        self.W_xf = nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=padding)
        self.W_hf = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding)
        self.W_cf = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding)

        self.W_xc = nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=padding)
        self.W_hc = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding)

        self.W_xo = nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=padding)
        self.W_ho = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding)
        self.W_co = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding)

        self.b_i = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))
        self.b_f = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))
        self.b_c = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))
        self.b_o = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))

    def forward(self, x, prev_state):
        if prev_state is None:
            batch_size, _, height, width = x.shape
            h_prev = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device)
            c_prev = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device)
        else:
            h_prev, c_prev = prev_state

        i_t = torch.sigmoid(self.W_xi(x) + self.W_hi(h_prev) + self.W_ci(c_prev) + self.b_i)
        f_t = torch.sigmoid(self.W_xf(x) + self.W_hf(h_prev) + self.W_cf(c_prev) + self.b_f)
        c_tilde = torch.tanh(self.W_xc(x) + self.W_hc(h_prev) + self.b_c)
        c_t = f_t * c_prev + i_t * c_tilde
        o_t = torch.sigmoid(self.W_xo(x) + self.W_ho(h_prev) + self.W_co(c_t) + self.b_o)
        h_t = o_t * torch.tanh(c_t)

        return h_t, (h_t, c_t)


class ResidualConvLSTMUnit(nn.Module):
    """ConvLSTM unit with residual connection"""

    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.conv_lstm = ImprovedConvLSTMUnit(in_channels, hidden_channels, kernel_size)

        if in_channels != hidden_channels:
            self.residual_conv = nn.Conv2d(in_channels, hidden_channels, 1)
        else:
            self.residual_conv = None

    def forward(self, x, prev_state):
        h_new, new_state = self.conv_lstm(x, prev_state)

        if self.residual_conv is not None:
            residual = self.residual_conv(x)
        else:
            residual = x

        h_new = h_new + residual
        return h_new, new_state