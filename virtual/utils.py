import torch
import torch.nn.functional as F
from torch.utils import data

class Dataset(data.Dataset):
    def __init__(self, args):
        self.n_states = args.n_states
        self.n_actions = args.n_actions

        self.states = []
        self.actions = []
        self.states_ = []

    def __len__(self):
        assert len(self.states) == len(self.actions)
        assert len(self.states) == len(self.states_)
        return len(self.states)

    def __getitem__(self, index):
        return self.states[index], self.actions[index], self.states_[index]

    def add_data(self, data):
        self.states += data[0]
        self.actions += data[1]
        self.states_ += data[2]

    def collate_fn(self, batch):
        states, actions, states_ = [], [], []
        for state, action, state_ in batch:
            # state
            x = torch.zeros(self.n_states)
            x[state] = 1
            states.append(x)
            # action
            x = torch.zeros(self.n_actions)
            x[action] = 1
            actions.append(x)
            # state_ (label)
            states_.append(state_)
        states = torch.stack(states, dim=0)
        actions = torch.stack(actions, dim=0)
        states_ = torch.tensor(states_, dtype=torch.long)
        return states, actions, states_

class Accuracy():
    def __init__(self):
        self.total = 0
        self.correct = 0

    def count(self, preds, targs):
        '''
        @param preds: batch_size x n_categories
        @param targs: batch_size
        '''
        # print(preds)
        # print(targs)
        preds = preds.argmax(dim=1)
        # print(preds)
        self.total += targs.shape[0]
        self.correct += preds.eq(targs).sum().item()
        # print(self.total)
        # print(self.correct)
        # assert False

    def report(self, reset=False):
        res = 1. * self.correct / self.total
        if (reset): self.reset()
        return res

    def reset(self):
        self.total = self.correct = 0
