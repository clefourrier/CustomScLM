import os, sys
import argparse
import functools
import random
from typing import Optional, Tuple

from dataclasses import dataclass
import numpy as np
import nltk
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AdamW
from transformers.utils import ModelOutput
import torch
from torch import nn


print = functools.partial(print, flush=True)


@dataclass
class CausalLMOutputWithActionsAndWords(ModelOutput):
    """
    Same as CausalLMOutputWithCrossAttentions, with added loss and logit storage
    """

    loss: Optional[torch.FloatTensor] = None  # overall loss
    word_prediction_loss: Optional[torch.FloatTensor] = None
    action_prediction_loss: Optional[torch.FloatTensor] = None
    word_logits: torch.FloatTensor = None
    action_logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


class Vocabulary(object):
    def __init__(self, symbols, pad_index):
        self.symbols = symbols
        self.pad_index = pad_index
        self.token2index = {}
        for i, symbol in enumerate(symbols):
            self.token2index[symbol] = i
        return

    def pad(self):
        return self.pad_index

    def convert_tokens_to_ids(self, token_input):
        if isinstance(token_input, str):
            return self.token2index[token_input]
        elif isinstance(token_input, list):
            return [self.token2index[token] for token in token_input]
        else:
            raise NotImplementedError

    def convert_ids_to_tokens(self, id_input):
        if isinstance(id_input, int):
            return self.symbols[id_input]
        elif isinstance(id_input, list):
            return [self.symbols[idx] for idx in id_input]
        else:
            raise NotImplementedError


class ScLMPreTrainedModel(GPT2PreTrainedModel):
    config_class = ScLMConfig
    load_tf_weights = load_tf_weights_in_gpt2
    base_model_prefix = "transformer"
    is_parallelizable = True
    supports_gradient_checkpointing = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)


class ScLM(ScLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            "gpt2", cache_dir=config.cache_dir
        )

        self.w_boundary_char = b"\xc4\xa0".decode()

        self.action_vocab = Vocabulary(config.action_ngram_list, 0)
        self.action_decoder = nn.Linear(768, len(self.action_vocab.symbols))

        if config.is_random_init:
            print("Initialize with random weights", file=sys.stderr)
            self.model = GPT2LMHeadModel(config)  # config subclasses GPT2Config
        else:
            self.model = GPT2LMHeadModel.from_pretrained(
                "gpt2", cache_dir=config.cache_dir
            )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        label_ids: Optional[torch.LongTensor] = None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(
                len(tokens_batch), 12, batch_max_len, batch_max_len, device=device
            )

        # todo: we need to add eos token in the tokenizer, to pad (pad to max_len) (same)
        # todo: inputs_ids and attention_mask will already contain correctly padded inputs, and transformers will take care of masking

        transformer_outputs = self.model(
            input_ids,
            # past_key_values=past_key_values,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            # position_ids=position_ids,
            # head_mask=head_mask,
            # inputs_embeds=inputs_embeds,
            # encoder_hidden_states=encoder_hidden_states,
            # encoder_attention_mask=encoder_attention_mask,
            labels=label_ids,
            # use_cache=use_cache,
            # output_attentions=output_attentions,
            output_hidden_states=True,
            # return_dict=return_dict,
        )

        word_prediction_loss = transformer_outputs[0]

        if scaffold_type == "next":
            action_ids_batch = torch.tensor(
                [
                    [self.action_vocab.token2index[a_ngram] for a_ngram in line[1]]
                    + [-100 for _ in range(batch_max_len - len(line[1]))]
                    for line in input_ids
                ]
            ).to(device)
        elif scaffold_type == "past":
            action_ids_batch = torch.tensor(
                [
                    [self.action_vocab.pad_index]
                    + [
                        self.action_vocab.token2index[a_ngram]
                        for a_ngram in line[1][:-1]
                    ]
                    + [-100 for _ in range(batch_max_len - len(line[1]))]
                    for line in input_ids
                ]
            ).to(device)
        else:
            raise NotImplementedError

        action_prediction_logits = self.action_decoder(
            transformer_outputs[-1][-1][:, :, :]
        )
        action_prediction_logits = action_prediction_logits.view(
            len(tokens_batch) * batch_max_len, -1
        )
        action_prediction_loss = torch.nn.CrossEntropyLoss(
            action_prediction_logits,
            action_ids_batch.view(len(tokens_batch) * batch_max_len, -1).squeeze(),
        )

        loss = (1 - ALPHA) * word_prediction_loss + ALPHA * action_prediction_loss

        # We might need to derive this class to create a custom output, if we want to display the word_prediction_loss and action_prediction_loss for validation
        return CausalLMOutputWithActionsAndWords(
            loss=loss,
            word_prediction_loss=word_prediction_loss,
            action_prediction_loss=action_prediction_loss,
            word_logits=transformer_outputs.logits,
            action_logits=action_prediction_logits,  # or create two values, most likely? (word_logits vs action_logits)
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    def get_surprisals(self, tokens, add_bos_token=True):
        surprisals = []
        for i in range(len(tokens)):
            token_id = self.tokenizer.convert_tokens_to_ids(tokens[i])
            if add_bos_token:
                # add BOS token
                prefix_tokens = [self.tokenizer.bos_token] + tokens[:i]
            else:
                if i == 0:
                    surprisals.append(0.0)
                    continue
                else:
                    prefix_tokens = tokens[:i]

            ids = self.tokenizer.convert_tokens_to_ids(prefix_tokens)
            input_ids = torch.tensor([ids]).to(device)
            output = self.model(input_ids)
            logits = output[0]
            next_token_logits = logits[:, -1, :].squeeze()
            log_probs = log_softmax(next_token_logits)
            surprisal = -log_probs[token_id] / np.log(2)
            surprisals.append(surprisal)
        return surprisals

    def get_word_ppl(self, sents, add_bos_token=True):
        nll_total = 0
        word_count = 0

        total_token_count = 0

        for sent in sents:
            words = sent.split()
            if add_bos_token:
                tokens = [self.tokenizer.bos_token] + self.tokenizer.tokenize(sent)
            else:
                tokens = self.tokenizer.tokenize(sent)
                if len(tokens) <= 1:
                    continue

            ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_ids = torch.tensor([ids]).to(device)  # batch size = 1

            loss = self.model(input_ids, labels=input_ids)[0].item()  # batch size = 1
            nll_total += loss * (len(tokens) - 1)
            word_count += len(words)

            total_token_count += len(tokens) - 1

        # print(nll_total, word_count, total_token_count)
        nll_avg = nll_total / word_count
        return np.exp(nll_avg)

    def generate(
        self,
        prompt,
        max_len=50,
        top_k=50,
        top_p=0.92,
        temperature=1,
        n_sample=1,
        device="cuda",
    ):
        """
        Sample from the model.
        """
        tokens = self.tokenizer.tokenize(prompt)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor([ids]).to(device)
        output_ids_batch = self.model.generate(
            input_ids,
            do_sample=True,
            max_length=max_len,
            pad_token_id=50256,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            num_return_sequences=n_sample,
        )
        samples = [
            self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            for output_ids in output_ids_batch
        ]
        return samples
