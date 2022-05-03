import numpy as np
import torch

from decision_transformer.training.trainer import Trainer
from torch.nn import functional as F

class SequenceTrainer(Trainer):

    def train_step(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask, actions_target = self.get_batch(self.batch_size)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, rtg[:,:,:-1], timesteps, attention_mask=attention_mask,
        )

        act_dim = actions_target.shape[-1]
        n_act = action_preds.shape[-1]
        n_agents = action_preds.shape[0]
        action_preds = action_preds.reshape(n_agents, -1, n_act)[attention_mask.unsqueeze(-1).repeat(n_agents, 1, 1, n_act).reshape(n_agents, -1, n_act) > 0]
        actions_target = actions_target.reshape(n_agents,-1, act_dim)[attention_mask.repeat(n_agents, 1, 1, act_dim).reshape(n_agents,-1) > 0]

        loss = F.cross_entropy(action_preds.reshape(-1, n_act), actions_target.reshape(-1).long())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = loss.detach().cpu().item()

        return loss.detach().cpu().item()