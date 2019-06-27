import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import virtual.models as models
from virtual.utils import Dataset, Distance

class VirtualEnv:

    def __init__(self, args, env):
        self.args = args
        self.env = env
        Model = getattr(models, args.model)

        # self.acc = Accuracy()
        self.dist = Distance(args, env)
        self.train_set = Dataset(args)
        self.dev_set = Dataset(args)
        # self.dataset = Dataset(args)
        self.model = Model(args)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=args.lr)

    def add_data(self, data, train_fraction=0.7):
        # self.dataset.add_data(data)
        indices = [i for i in range(len(data[0]))]
        random.shuffle(indices)
        train_data, dev_data = [], []
        for j in range(3):
        	data[j] = [data[j][idx] for idx in indices]
        	train_data.append(data[j][:int(train_fraction * len(data[j]))])
        	dev_data.append(data[j][int(train_fraction * len(data[j])):])
        self.train_set.add_data(train_data)
        self.dev_set.add_data(dev_data)

    def train(self):
        step = 0
        #print("size of dataset is ", len(self.train_set))
        for epoch in range(self.args.n_epochs):

            data_loader = DataLoader(
                self.train_set,
                self.args.batch_size,
                shuffle=True,
                collate_fn=self.train_set.collate_fn
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

                # self.acc.count(preds, states_)
                self.dist.count(states, actions, preds, states_)
                if self.args.log_every > 0 and step % self.args.log_every == 0:
                    # print('Epoch %i step %i: train acc %f, loss %f' % (
                        # epoch, step, self.acc.report(reset=True), loss.item()))
                    print('Epoch %i step %i: train dist %f, loss %f' % (
                        epoch, step, self.dist.report(reset=True), loss.item()))
                if self.args.save_every > 0 and step % self.args.save_every == 0:
                    torch.save(self.model.state_dict(),
                        os.path.join(self.args.run_dir, 'params_%i.model' % step))
                if self.args.eval_every > 0 and step % self.args.eval_every == 0:
                    print('Epoch %i step %i: ' % (epoch, step), end='')
                    self.evaluate()

    def predict(self, states, actions):
        assert len(states) == len(actions)
        batches = [[states[i], actions[i], states[i]] for i in range(len(states))]
        states, actions, _ = self.dev_set.collate_fn(batches)
        
        self.model.eval()
        preds = self.model(states, actions)
        #print(preds)
        preds = preds.tolist()
        return preds

    def evaluate(self):
        self.model.eval()
        # acc = Accuracy()
        dist = Distance(self.args, self.env)
        data_loader = DataLoader(
            self.dev_set,
            self.args.batch_size,
            collate_fn=self.dev_set.collate_fn
        )
        for batch in data_loader:
            states, actions, states_ = batch
            preds = self.model(states, actions)
            # acc.count(preds, states_)
            dist.count(states, actions, preds, states_)
        # print('eval acc %f' % acc.report())
        print('eval dist %f' % dist.report())


if __name__ == '__main__':

    class Argument():
        def __init__(self):
            self.model = 'MLP'
            self.n_states = 4
            self.n_actions = 4
            self.d_hidden = 100
            self.n_layers = 1
            self.dropout = 0.5
            self.lr = 1e-4
            self.n_epochs = 100
            self.batch_size = 2
            self.log_every = 10
            self.save_every = 0
            self.eval_every = 10
    args = Argument()
    
    states = [0, 0, 1, 1, 0, 0, 1, 1]
    actions = [0, 1, 0, 1, 0, 1, 0, 1]
    states_ = [0, 2, 3, 0, 0, 2, 3, 0]
    data = [states, actions, states_]

    env = VirtualEnv(args)
    env.add_data(data)
    env.train()
    print(env.predict(states, actions))
