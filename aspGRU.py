import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.jit as jit
from typing import List, Tuple
from torch.nn import init
from torch import Tensor
import math

'''
Some helper classes for writing custom TorchScript LSTMs.
Goals:
- Classes are easy to read, use, and extend
- Performance of custom LSTMs approach fused-kernel-levels of speed.
A few notes about features we could add to clean up the below code:
- Support enumerate with nn.ModuleList:
  https://github.com/pytorch/pytorch/issues/14471
- Support enumerate/zip with lists:
  https://github.com/pytorch/pytorch/issues/15952
- Support overriding of class methods:
  https://github.com/pytorch/pytorch/issues/10733
- Support passing around user-defined namedtuple types for readability
- Support slicing w/ range. It enables reversing lists easily.
  https://github.com/pytorch/pytorch/issues/10774
- Multiline type annotations. List[List[Tuple[Tensor,Tensor]]] is verbose
  https://github.com/pytorch/pytorch/pull/14922
'''


def reverse(x):
    # type: (Tensor) -> Tensor
    return torch.flip(x, [0])


class AspectGRUCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size, asp_size):
        super(AspectGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(3 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(3 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(3 * hidden_size))
        self.bias_hh = Parameter(torch.randn(3 * hidden_size))
        self.weight = Parameter(torch.randn(hidden_size, hidden_size + input_size))
        self.bias_weight = Parameter(torch.randn(hidden_size))
        self.w1 = nn.Linear(input_size, hidden_size)
        self.w2 = nn.Linear(input_size, hidden_size)
        self.wa = nn.Linear(asp_size, hidden_size)
        self.whg = nn.Linear(hidden_size, hidden_size)

    @jit.script_method
    def forward(self, input, state, asp_hidden):
        # type: (Tensor, Tensor,Tensor) -> Tensor
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(state, self.weight_hh.t()) + self.bias_hh)
        updategate, resetgate, l = gates.chunk(3, 1)

        updategate = torch.sigmoid(updategate)
        resetgate = torch.sigmoid(resetgate)
        g = torch.relu(self.wa(asp_hidden) + self.whg(state))

        h2x = self.w2(input)
        h1x = self.w1(input)

        next_state = resetgate * state
        next_state = torch.cat((next_state, input), dim=1)
        next_state = torch.mm(next_state, self.weight.t()) + self.bias_weight
        next_state = torch.tanh(next_state) + g * h2x + l * h1x
        next_state = (1 - updategate) * state + updategate * next_state

        return next_state


class AspectGRULayer(jit.ScriptModule):
    def __init__(self, input_sz, hidden_sz, asp_size):
        super(AspectGRULayer, self).__init__()
        self.cell = AspectGRUCell(input_sz, hidden_sz, asp_size)

    @jit.script_method
    def forward(self, input, state, asp_hidden):
        # type: (Tensor, Tensor,Tensor) -> Tuple[Tensor, Tensor]
        inputs = input.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            state = self.cell(inputs[i], state, asp_hidden)  # batch x hidden_size
            outputs += [state]
        return torch.stack(outputs), state  # [seqlen, batch, hidden]   [batch x hidden_size]


class ReverseAspectGRULayer(jit.ScriptModule):
    def __init__(self, input_sz, hidden_sz, asp_size):
        super(ReverseAspectGRULayer, self).__init__()
        self.cell = AspectGRUCell(input_sz, hidden_sz, asp_size)

    @jit.script_method
    def forward(self, input, state, asp_hidden):
        # type: (Tensor, Tensor,Tensor) -> Tuple[Tensor, Tensor]
        inputs = reverse(input).unbind(0)
        outputs = jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            state = self.cell(inputs[i], state, asp_hidden)
            outputs += [state]
        return reverse(torch.stack(outputs, dim=0)), state  # [seqlen, batch, hidden]   [batch x hidden_size]


class BiAspectGRULayer(jit.ScriptModule):
    def __init__(self, input_sz, hidden_sz, asp_size):
        super(BiAspectGRULayer, self).__init__()
        self.directions = nn.ModuleList([
            AspectGRULayer(input_sz, hidden_sz, asp_size),
            ReverseAspectGRULayer(input_sz, hidden_sz, asp_size),
        ])
        stdv = 1.0 / math.sqrt(hidden_sz)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    @jit.script_method
    def forward(self, input, state, asp_hidden):
        # type: (Tensor,Tensor,Tensor) -> Tuple[Tensor, Tensor]
        # List[LSTMState]: [forward LSTMState, backward LSTMState]
        outputs = jit.annotate(List[Tensor], [])
        output_states = jit.annotate(List[Tensor], [])
        input = input.transpose(1, 0).contiguous()
        out_state = state
        for direction in self.directions:
            out, out_state = direction(input, out_state, asp_hidden)  # [seqlen, batch, hidden]   [batch x hidden_size]
            outputs += [out]
            output_states += [out_state]
        # [seqlen, batch, num_direction*hidden_sz]   [num_direction, batch , hidden_size]
        return torch.cat(outputs, -1).transpose(1, 0).contiguous(), torch.stack(output_states, dim=0).contiguous()
