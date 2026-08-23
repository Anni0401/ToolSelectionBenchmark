import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, KLDivLoss



# ---------------------------------------
# >> REPLUG

def compute_loss(Q, Pr, kl_div):
    """
    Computes KL(Pr||Q)
    (Q and P are inverted in the function parameters, because it's how pytorch wants them)
    """

    # To avoid underflow issues when computing this quantity, this loss expects the argument input in the log-space.
    
    #Q_log = torch.log(Q)
    # divergence = kl_div(Q_log, Pr).sum(-1)
    
    Pr_log = torch.log(Pr)
    divergence = kl_div(Pr_log, Q).sum(-1)
    
    loss = divergence.mean()
    return loss

def compute_perplexity(output, labels, pad_token_id):
    logits = output["logits"]  # [bs, seq_len, vocab_size]

    # Ensure labels are within the valid range of indices
    assert torch.all(labels < logits.size(-1)), "Labels contain indices out of range for logits"

    #log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # [bs, seq_len, vocab_size]
    log_probs = torch.nn.functional.softmax(logits, dim=-1)

    # Apply gather to get the log probabilities for the correct labels
    x = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))  # [bs, seq_len, 1]
    x = x.squeeze(-1)  # [bs, seq_len]

    # Create a mask for non-pad tokens
    mask = (labels != pad_token_id).float()  # [bs, seq_len]

    # Compute the mean log probability only for non-pad tokens
    masked_x = x * mask  # [bs, seq_len]
    mean_log_prob = masked_x.sum() / mask.sum()  # scalar

    return -mean_log_prob

# def get_perplexity(outputs,
#                    input_ids,
#                    attention_mask):
#     # Get the loss for each element in the batch
#     loss = outputs.loss_per_sample if hasattr(outputs, 'loss_per_sample') else outputs.loss.unsqueeze(0)

#     # Count the number of non-padding tokens for each sample
#     non_pad_tokens = attention_mask.sum(dim=1)

#     # Compute perplexity for each sample
#     #ppl = torch.exp(loss * input_ids.size(1) / non_pad_tokens)
#     ppl = loss * input_ids.size(1) / non_pad_tokens

#     return ppl

from torch.nn import CrossEntropyLoss
cross_entropy = CrossEntropyLoss(reduction='none')

def get_perplexity(outputs, 
                   input_ids,
                   attention_mask,
                   padding_token_ids : int = -100):
    """
    From the inference model's outputs and the labels, compute the perplexity of each example
    """
    labels=input_ids
    logits = outputs["logits"]
    shift_labels = labels[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    
    elem_wise_loss = cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss_sum_per_sample = elem_wise_loss.view(shift_logits.size(0), shift_logits.size(1)).sum(dim=1)
    #num_elems_per_sample = torch.sum(shift_labels.ne(padding_token_ids), dim=1)
    num_elems_per_sample = torch.sum(shift_labels.ne(-100), dim=1)
    loss_per_sample = loss_sum_per_sample / num_elems_per_sample

    return -loss_per_sample

def compute_Pr(similarities, axis):
    return F.softmax(similarities, dim=axis)


# ----------------------------------------
# >> ORPO

# ORPO Authors: Jiwoo Hong, Noah Lee, and James Thorne
# Official code: https://github.com/xfactlab/orpo
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        label_pad_token_id: int = -100,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
            label_pad_token_id: The label pad token id.
            is_encoder_decoder: Whether the model is an encoder-decoder model.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)


def odds_ratio_loss(positive_retr_log_prob, 
                    negative_retr_log_prob, 
                    beta : float = 0.1):
    """
    Return the odd ration between input posiitve and negatative probabilities
    """
                
    # log ration reformulated as difference to enhance statbility
    log_odds = (positive_retr_log_prob - negative_retr_log_prob) - (
        torch.log1p(-torch.exp(positive_retr_log_prob)) - torch.log1p(-torch.exp(negative_retr_log_prob))
    )

    sig_ratio = F.sigmoid(log_odds)
    ratio = torch.log(sig_ratio)
    losses = beta * ratio

    positive_rewards = beta * positive_retr_log_prob.detach()
    negative_rewards = beta * negative_retr_log_prob.detach()

    return losses, positive_rewards, negative_rewards, torch.mean(ratio).item(), torch.mean(log_odds).item()