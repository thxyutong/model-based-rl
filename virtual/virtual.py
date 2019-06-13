import torch
import torch.nn.functional as F
from torch.utils import data

from . import models

class VirtualEnv:

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
                x[0, state] = 1
                states.append(x)
                # action
                x = torch.zeros(self.n_actions)
                x[0, action] = 1
                actions.append(x)
                # state_ (label)
                states_.append(state_)
            states = torch.stack(states, dim=0)
            actions = torch.stack(actions, dim=0)
            states_ = torch.tensor(states_, dtype=long)
            return statse, actions, states_


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
            self.correct += preds.eq(targs).sum()

        def report(self, reset=False):
            res = 1. * self.correct / self.total
            if (reset): self.reset()
            return res

        def reset(self):
            self.total = self.correct = 0



    def __init__(self, args):
        self.args = args
        Model = getattr(models, args.model)

        self.acc = Accuracy()
        self.dataset = Dataset()
        self.model = Model(args)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def add_data(self, data):
        self.dataset.add_data(data)

    def train(self):
        step = 0
        for epoch in range(self.args.n_epochs):

            data_loader = data.DataLoader(
                self.dataset,
                self.args.batch_size,
                shuffle=True,
                collate_fn=self.dataset.collate_fn
            )

            for batch in data_loader:
                states, actions, states_ = batch

                self.model.train()
                self.optimizer.zero_grad()

                preds = self.model(states, actions)
                loss = F.cross_entropy(preds, states_)
                loss.backward()
                self.optimizer.step()
                step += 1

                self.acc.count(preds, states_)
                if args.log_every > 0 and step % args.log_every == 0:
                    print('train acc %f' % self.acc.report(reset=True))
                if args.eval_every > 0 and step % args.eval_every == 0:
                    self.evaluate()
                if args.save_every > 0 and step % args.save_every == 0:
                    torch.save(self.model.state_dict(),
                        os.path.join(args.run_dir, 'params_%i.model' % step))

    def evaluate(self):
        self.model.eval()
        acc = Accuracy()
        data_loader = data.DataLoader(
            self.dataset,
            self.args.batch_size,
            collate_fn=self.dataset.collate_fn
        )
        for batch in data_loader:
            states, actions, states_ = batch
            preds = self.model(states, actions)
            acc.count(preds, states_)
        print('eval acc %f' % acc.report())