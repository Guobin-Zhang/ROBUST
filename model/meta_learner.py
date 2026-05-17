import torch
import torch.nn as nn
import copy

class MAMLLearner:
    def __init__(self, model, inner_lr=0.01, outer_lr=0.001, inner_steps=5):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=outer_lr)

    def inner_loop(self, task_data, task_payload):
        fast_weights = {name: param.clone() for name, param in self.model.named_parameters()}
        for _ in range(self.inner_steps):
            stego = self.model(task_data, task_payload, use_fast_weights=fast_weights)
            loss = nn.functional.mse_loss(stego, task_data)
            grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
            for (name, _), grad in zip(fast_weights.items(), grads):
                fast_weights[name] = fast_weights[name] - self.inner_lr * grad
        return fast_weights

    def outer_loop(self, task_batches):
        self.meta_optimizer.zero_grad()
        meta_loss = 0.0
        for task_data, task_payload in task_batches:
            fast_weights = self.inner_loop(task_data, task_payload)
            stego = self.model(task_data, task_payload, use_fast_weights=fast_weights)
            meta_loss += nn.functional.mse_loss(stego, task_data)
        meta_loss /= len(task_batches)
        meta_loss.backward()
        self.meta_optimizer.step()
        return meta_loss.item()