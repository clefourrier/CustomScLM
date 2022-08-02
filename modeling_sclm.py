import os
import sys
import argparse
import random
import numpy as np
import nltk
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AdamW
import torch
from torch import nn

import functools
import utils
print = functools.partial(print, flush=True)


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


class ScLMPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
        Copy of GPT2PretrainedModel
    """
    config_class = GPT2Config
    pretrained_model_archive_map = GPT2_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_gpt2
    base_model_prefix = "transformer"

    def __init__(self, *inputs, **kwargs):
        super(GPT2PreTrainedModel, self).__init__(*inputs, **kwargs)

        # to check !!!!
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            config.model_name, 
            cache_dir=config.cache_dir
        )

        self.action_vocab = Vocabulary(config.action_ngram_list, 0)

        self.w_boundary_char = b'\xc4\xa0'.decode()
        self.action_decoder = nn.Linear(768, len(self.action_vocab.symbols)).to(config.device)


    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            

class ScLM(ScLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # Load pretrained tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(config.model_name, cache_dir=config.cache_dir)

        if config.is_random_init:
            print('Initialize with random weights', file=sys.stderr)
            gpt2_config = GPT2Config(len(self.tokenizer))
            self.model = GPT2LMHeadModel(gpt2_config).to(config.device)
        else:
            print('Initialize with pretrained weights', file=sys.stderr)
            self.model = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir=config.cache_dir).to(config.device)

        self.action_vocab = Vocabulary(config.action_ngram_list, 0)

        self.w_boundary_char = b'\xc4\xa0'.decode()
        self.action_decoder = nn.Linear(768, len(self.action_vocab.symbols)).to(config.device)

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
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(len(tokens_batch), 12, batch_max_len, batch_max_len, device=device)

        # todo: we need to add eos token in the tokenizer, to pad (pad to max_len) (same)
        # todo: inputs_ids and attention_mask will already contain correctly padded inputs, and transformers will take care of masking

        transformer_outputs = self.model(
            input_ids,
            #past_key_values=past_key_values,
            attention_mask=attention_mask,
            #token_type_ids=token_type_ids,
            #position_ids=position_ids,
            #head_mask=head_mask,
            #inputs_embeds=inputs_embeds,
            #encoder_hidden_states=encoder_hidden_states,
            #encoder_attention_mask=encoder_attention_mask,
            labels=label_ids,
            #use_cache=use_cache,
            #output_attentions=output_attentions,
            output_hidden_states=True,
            #return_dict=return_dict,
        )

        word_prediction_loss = transformer_outputs[0]

        if scaffold_type == 'next':
            action_ids_batch = torch.tensor([[self.action_vocab.token2index[a_ngram] for a_ngram in line[1]] + [-100 for _ in range(batch_max_len - len(line[1]))] for line in input_ids]).to(device)
        elif scaffold_type == 'past':
            action_ids_batch = torch.tensor([[self.action_vocab.pad_index] + [self.action_vocab.token2index[a_ngram] for a_ngram in line[1][:-1]] + [-100 for _ in range(batch_max_len - len(line[1]))] for line in input_ids]).to(device)
        else:
            raise NotImplementedError

        action_prediction_logits = self.action_decoder(transformer_outputs[-1][-1][:, :, :])
        action_prediction_logits = action_prediction_logits.view(len(tokens_batch)*batch_max_len, -1)
        action_prediction_loss = torch.nn.CrossEntropyLoss(action_prediction_logits, action_ids_batch.view(len(tokens_batch)*batch_max_len, -1).squeeze())

        loss = (1 - ALPHA) * word_prediction_loss + ALPHA * action_prediction_loss

        # We might need to derive this class to create a custom output, if we want to display the word_prediction_loss and action_prediction_loss for validation
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=transformer_outputs.logits, # or action_prediction_logits ? # or create two values, most likely? (word_logits vs action_logits)
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )


    def get_batch_loss(self, line_batch, scaffold_type, add_eos_token=False):
        # Original function
        """
        Assume each line of the batch input contains tokenized sentence paired with action ngram sequence.
        """
        if add_eos_token:
            tokens_batch = [line[0] + [self.tokenizer.bos_token] for line in line_batch]
        else:
            tokens_batch = [line[0] for line in line_batch]

        batch_max_len = np.max([len(tokens) for tokens in tokens_batch])
        tokens_padded_batch = [tokens + [self.tokenizer.bos_token for _ in range(batch_max_len - len(tokens))] for tokens in tokens_batch]

        # Mask padded tokens
        attention_mask = torch.ones(len(tokens_batch), 12, batch_max_len, batch_max_len).to(device)
        for b_idx, tokens in enumerate(tokens_batch):
            attention_mask[b_idx, :, :, len(tokens):] = 0

        input_ids_padded_batch = [self.tokenizer.convert_tokens_to_ids(tokens_padded) for tokens_padded in tokens_padded_batch]
        input_ids = torch.tensor(input_ids_padded_batch).to(device)

        label_ids_padded_batch = [self.tokenizer.convert_tokens_to_ids(tokens) + [-100 for _ in range(batch_max_len - len(tokens))] for tokens in tokens_batch]
        label_ids = torch.tensor(label_ids_padded_batch).to(device)

        output = self.model(input_ids, labels=label_ids, attention_mask=attention_mask, output_hidden_states=True)

        word_prediction_loss = output[0]

        if scaffold_type == 'next':
            action_ids_batch = torch.tensor([[self.action_vocab.token2index[a_ngram] for a_ngram in line[1]] + [-100 for _ in range(batch_max_len - len(line[1]))] for line in line_batch]).to(device)
        elif scaffold_type == 'past':
            action_ids_batch = torch.tensor([[self.action_vocab.pad_index] + [self.action_vocab.token2index[a_ngram] for a_ngram in line[1][:-1]] + [-100 for _ in range(batch_max_len - len(line[1]))] for line in line_batch]).to(device)
        else:
            raise NotImplementedError

        action_prediction_logits = self.action_decoder(output[-1][-1][:, :, :])
        action_prediction_logits = action_prediction_logits.view(len(tokens_batch)*batch_max_len, -1)
        action_prediction_loss = torch.nn.CrossEntropyLoss(action_prediction_logits, action_ids_batch.view(len(tokens_batch)*batch_max_len, -1).squeeze())

        return word_prediction_loss, action_prediction_loss

    def get_validation_loss(self, dev_lines, scaffold_type, word_prediction_loss_only=False):
        # To get this in Trainer we need to extend it to display those metrics
        if word_prediction_loss_only:
            # Only evaluate word prediction loss
            loss_sum = 0
            token_count = 0

            for line in dev_lines:
                tokens = line[0]
                ids = self.tokenizer.convert_tokens_to_ids(tokens)
                input_ids = torch.tensor([ids]).to(device) # batch size = 1

                loss = self.model(input_ids, labels=input_ids)[0].item()
                loss_sum += loss*(len(tokens)-1)
                token_count += len(tokens) - 1

            loss_avg = loss_sum/token_count

            return loss_avg

        else:
            word_loss_sum = 0
            action_loss_sum = 0
            word_token_count = 0
            action_token_count = 0

            for line_batch in get_batches(dev_lines, args.batch_size):
                n_word_token = np.sum([len(word_tokens) - 1 for word_tokens, _ in line_batch])
                n_action_token = n_word_token + len(line_batch)

                word_prediction_loss, action_prediction_loss = self.get_batch_loss(line_batch, scaffold_type, add_eos_token=False)

                word_loss_sum += word_prediction_loss.item()*n_word_token
                action_loss_sum += action_prediction_loss.item()*n_action_token

                word_token_count += n_word_token
                action_token_count += n_action_token

            word_loss_avg = word_loss_sum/word_token_count
            action_loss_avg = action_loss_sum/action_token_count
            loss_avg = (1 - ALPHA)*word_loss_avg + ALPHA*action_loss_avg

            return loss_avg, word_loss_avg, action_loss_avg


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
            surprisal = -log_probs[token_id]/np.log(2)
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
            input_ids = torch.tensor([ids]).to(device) # batch size = 1

            loss = self.model(input_ids, labels=input_ids)[0].item() # batch size = 1
            nll_total += loss*(len(tokens)-1)
            word_count += len(words)

            total_token_count += len(tokens)-1

        #print(nll_total, word_count, total_token_count)
        nll_avg = nll_total/word_count
        return np.exp(nll_avg)

    def generate(self, prompt, max_len=50, top_k=50, top_p=0.92, temperature=1, n_sample=1, device='cuda'):
        """
        Sample from the model.
        """
        tokens = self.tokenizer.tokenize(prompt)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor([ids]).to(device)
        output_ids_batch = self.model.generate(input_ids, do_sample=True, max_length=max_len, pad_token_id=50256,
                                               top_k=top_k, top_p=top_p, temperature=temperature, num_return_sequences=n_sample)
        samples = [self.tokenizer.decode(output_ids, skip_special_tokens=True).strip() for output_ids in output_ids_batch]
        return samples
