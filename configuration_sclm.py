# coding=utf-8
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

from transformers import GPT2Config
from transformers.utils import logging


logger = logging.get_logger(__name__)


# TODO this is not done at all!!!!
class ScLMConfig(GPT2Config):
    def __init__(self, is_random_init, action_ngram_list, device='cuda', model_name='gpt2', cache_dir='pretrained/gpt2'):

        if is_random_init:
            print('Initialize with random weights', file=sys.stderr)
            config = GPT2Config(len(self.tokenizer))
            self.model = GPT2LMHeadModel(config).to(device)
        else:
            print('Initialize with pretrained weights', file=sys.stderr)
            self.model = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir=cache_dir).to(device)

        self.action_vocab = Vocabulary(action_ngram_list, 0)

        self.w_boundary_char = b'\xc4\xa0'.decode()
        self.model.action_decoder = torch.nn.Linear(768, len(self.action_vocab.symbols)).to(device)
