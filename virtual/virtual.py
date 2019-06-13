import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import models
from utils import Dataset, Accuracy

class VirtualEnv:

    def __init__(self, args):
        self.args = args
        Model = getattr(models, args.model)

        self.acc = Accuracy()
        self.dataset = Dataset(args)
        self.model = Model(args)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=args.lr)

    def add_data(self, data):
        self.dataset.add_data(data)

    def train(self):
        step = 0
        for epoch in range(self.args.n_epochs):

            data_loader = DataLoader(
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
                    print('Epoch %i step %i: train acc %f, loss %f' % (
                        epoch, step, self.acc.report(reset=True), loss.item()))
                if args.save_every > 0 and step % args.save_every == 0:
                    torch.save(self.model.state_dict(),
                        os.path.join(args.run_dir, 'params_%i.model' % step))
                # if args.eval_every > 0 and step % args.eval_every == 0:
                #     self.evaluate()

    # def evaluate(self):
    #     self.model.eval()
    #     acc = Accuracy()
    #     data_loader = data.DataLoader(
    #         self.dataset,
    #         self.args.batch_size,
    #         collate_fn=self.dataset.collate_fn
    #     )
    #     for batch in data_loader:
    #         states, actions, states_ = batch
    #         preds = self.model(states, actions)
    #         acc.count(preds, states_)
    #     print('eval acc %f' % acc.report())


if __name__ == '__main__':

    class Argument():
        def __init__(self):
            self.model = 'MLP'
            self.n_states = 2
            self.n_actions = 2
            self.d_hidden = 10
            self.n_layers = 1
            self.dropout = 0.5
            self.lr = 1e-4
            self.n_epochs = 10000
            self.batch_size = 2
            self.log_every = 10
            self.save_every = 0
    args = Argument()
    
    states = [0, 0, 1, 1]
    actions = [0, 1, 0, 1]
    states_ = [0, 1, 1, 0]
    data = [states, actions, states_]

    env = VirtualEnv(args)
    env.add_data(data)
    env.train()