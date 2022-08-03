import time
from typing import Optional, List, Dict

from datasets import Dataset, load_metric

import evaluate

import numpy as np

import torch

from transformers import PreTrainedTokenizerFast, TrainingArguments, Trainer
from transformers.trainer import speed_metrics


class TrainerWithEvalLoss(Trainer):
    """ Custom trainer which displays training accuracy"""

    def init(
        self,
        model,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
    ):
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
        )

    def evaluate(self, ignore_keys: Optional[List[str]] = None) -> Dict[str, float]:

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        # On train set
        train_dataloader = self.get_train_dataloader()
        start_time_train = time.time()
        train_output = self.prediction_loop(
            train_dataloader,
            description="Training",
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix="train",
        )
        train_n_samples = len(self.train_dataset)
        train_output.metrics.update(
            speed_metrics("train", start_time_train, train_n_samples)
        )
        self.log(train_output.metrics)
        # print(train_output.metrics, train_n_samples, train_dataloader.dataset)

        # On val set
        eval_dataloader = self.get_eval_dataloader()
        start_time_val = time.time()
        eval_output = self.prediction_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix="eval",
        )

        eval_n_samples = len(self.eval_dataset)
        eval_output.metrics.update(
            speed_metrics("eval", start_time_val, eval_n_samples)
        )
        self.log(eval_output.metrics)
        # print(eval_output.metrics, eval_n_samples, eval_dataloader.dataset)

        eval_output.metrics["eval_loss"] = "No log"
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, eval_output.metrics
        )
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, train_output.metrics
        )

        self._memory_tracker.stop_and_update_metrics(train_output.metrics)
        self._memory_tracker.stop_and_update_metrics(eval_output.metrics)

        metrics_dict = {
            "Training metrics": train_output.metrics,
            "Validation metrics": eval_output.metrics,
        }

        return metrics_dict
