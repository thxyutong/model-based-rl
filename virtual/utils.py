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
            states.append(state)
            actions.append(action)
            states_.append(state_)
        states = torch.tensor(states, dtype=torch.long)
        actions = torch.tensor(actions, dtype=torch.long)
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
        preds = preds.argmax(dim=1)
        self.total += targs.shape[0]
        self.correct += preds.eq(targs).sum().item()

    def report(self, reset=False):
        res = 1. * self.correct / self.total
        if (reset): self.reset()
        return res

    def reset(self):
        self.total = self.correct = 0
        
class Distance():
    def __init__(self, args):
        self.n_states = args.n_states
        self.pred_count = dict()
        self.real_count = dict()
        
    def count(self, states, actions, preds, states_):
        assert states.shape[0] == actions.shape[0]
        assert states.shape[0] == states_.shape[0]
        assert states.shape[0] == preds.shape[0]
        
        for i in range(states.shape[0]):
            state = states[i]
            action = actions[i]
            state_ = states_[i]
            pair = (state.item(), action.item())
            
            if pair not in self.pred_count:
                self.pred_count[pair] = torch.zeros(self.n_states)
                self.real_count[pair] = torch.zeros(self.n_states)
                
            self.pred_count[pair] += preds[i]
            self.real_count[pair][state_] += 1
            
    def report(self, reset=False):
        cnt = 0
        dist = 0.0
        for key in self.pred_count.keys():
            pred = self.pred_count[key]
            real = self.real_count[key]
            cur_cnt = int(real.sum().item())
            cnt += cur_cnt
            pred /= cur_cnt
            real /= cur_cnt
            dist += 0.5 * (real - pred).abs().sum()
        res = dist / cnt
        if (reset): self.reset()
        return res
        
    def reset(self):
        self.pred_count = dict()
        self.real_count = dict()
