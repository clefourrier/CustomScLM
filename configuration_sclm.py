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


# TODO check
class ScLMConfig(GPT2Config):
    def __init__(
        self,
        is_random_init,
        action_ngram_list,
        model_name="gpt2",
        cache_dir="pretrained/gpt2",
        **kwargs
    ):
        self.is_random_init = is_random_init
        self.action_ngram_list = action_ngram_list
        self.model_name = model_name
        self.cache_dir = cache_dir

        super(GPT2Config, self).__init__(**kwargs)
