import torch
from torch import nn

from modules import TraditionalLayer

class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_classes = args.n_states
        self.state_embedding = nn.Linear(args.n_states, args.d_hidden)
        self.action_embedding = nn.Linear(args.n_actions, args.d_hidden)
        self.output = nn.Sequential(
          *[TraditionalLayer(
                args.d_hidden * 2,
                args.d_hidden * 2,
                dropout=args.dropout
            ) for _ in range(args.n_layers)],
            nn.Linear(args.d_hidden * 2, self.n_classes)
        )

    def forward(self, states, actions):
        '''
        @param states:  batch_size x n_states
        @param actions: batch_size x n_actions
        @return:        batch_size x n_classes(n_states)
        '''
        states = self.state_embedding(states)    # batch_size x d_hidden
        actions = self.action_embedding(actions) # batch_size x d_hidden
        x = torch.cat([states, actions], dim=1)  # batch_size x d_hidden*2
        x = self.output(x) # batch_size x n_classes
        return torch.softmax(x, dim=1)
