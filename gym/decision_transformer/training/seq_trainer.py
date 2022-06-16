import numpy as np
import torch

from decision_transformer.training.trainer import Trainer
from torch.nn import functional as F

class SequenceTrainer(Trainer):

    def train_step(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        action_target = torch.clone(actions)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )

        act_dim = action_target.shape[2]
        n_act = action_preds.shape[2]
        n_actions = action_preds.shape[-1]
        action_preds = action_preds.reshape(-1, n_act)[attention_mask.unsqueeze(-1).repeat(1, 1, 1, n_actions).reshape(-1, n_act) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        
        action_preds = action_preds.reshape(-1, n_act)
        action_target = action_target.reshape(-1).long()

        loss = F.cross_entropy(action_preds, action_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = loss.detach().cpu().item()

        return loss.detach().cpu().item()
