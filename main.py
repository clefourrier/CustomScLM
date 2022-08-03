from datasets import load_dataset, Dataset
from torch.optim import AdamW
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import TrainingArguments, Trainer, enable_determinism

from configuration_sclm import ScLMConfig  # todo, useless for now
from modeling_sclm import ScLM
from utils import TrainerWithEvalLoss


def is_nonterminal(token):
    """
    Check if a token is a nonterminal action or word piece token.
    """
    if (token.startswith("NT(") and token.endswith(")")) or token == "REDUCE()":
        return True
    else:
        return False


def load_sents(path):
    # todo: save dataset on hugging face to load it here? or use as preprocessing step
    with open(path) as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line != ""]
    return lines


def load_data(path, tokenizer, BOS_token=None):
    # todo: add dataloader here, with correct tokenizer map
    with open(path) as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line != ""]

    data = []

    for line in lines:
        tokens = line.split()
        action_ngrams = []
        words = []
        action_seq = []
        for token in tokens:
            if is_nonterminal(token):
                action_seq.append(token)
            else:
                if action_seq != []:
                    action_ngrams.append("_".join(action_seq))
                    action_seq = []
                else:
                    action_ngrams.append("_")
                words.append(token)

        action_ngrams.append(
            "_".join(action_seq)
        )  # add the action ngram that comes after the last word

        sent = " ".join(words)
        word_pieces = tokenizer.tokenize(sent)

        combined = ""
        n_piece = 0
        word_index = 0
        action_ngram_seq = []

        for piece in word_pieces:
            if piece.startswith(w_boundary_char):
                combined += piece[1:]
            else:
                combined += piece
            n_piece += 1
            if combined == words[word_index]:
                action_ngram_seq += [action_ngrams[word_index]] + [
                    "_" for _ in range(n_piece - 1)
                ]
                combined = ""
                n_piece = 0
                word_index += 1
        assert combined == ""
        assert word_index == len(words)

        action_ngram_seq.append(action_ngrams[-1])

        assert len(word_pieces) == (len(action_ngram_seq) - 1)

        if BOS_token is None:
            data.append([word_pieces, action_ngram_seq])
        else:
            data.append([[BOS_token] + word_pieces, action_ngram_seq])

    return data  # return dataset object


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="Path to training data.")
    parser.add_argument("--dev_data", type=str, help="Path to validation data.")
    parser.add_argument("--test_data", type=str, help="Path to test data.")
    parser.add_argument(
        "--fpath", type=str, help="File path for estimating surprisals."
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Hyerparameter in (0, 1) for weighting the structure prediction loss against the word prediction loss. Default is 0.5.",
    )
    parser.add_argument(
        "--scaffold_type", type=str, help="Type of scaffold. (next, past)"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--report",
        type=int,
        default=1000,
        help="Frequency of report training status after number of training batches.",
    )
    parser.add_argument(
        "--valid_every",
        type=int,
        default=None,
        help="Frequency of validating and saving model parameters after number of training batches.",
    )
    parser.add_argument(
        "--sample_every",
        type=int,
        default=10000,
        help="Frequency of generating samples from the model during training.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to train the model."
    )
    parser.add_argument(
        "--do_test", action="store_true", help="Whether to test the model."
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="Whether to use the model for surprisal estimation.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path of the model to be trained and saved.",
    )
    parser.add_argument(
        "--restore_from",
        type=str,
        default=None,
        help="Path to the trained model checkpoint. Will use the pretrained model if path not specified.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=5, help="Size of a training batch."
    )
    parser.add_argument(
        "--early_stopping_threshold",
        type=int,
        default=2,
        help="Threshold for early stopping.",
    )
    parser.add_argument(
        "--random_init",
        action="store_true",
        help="Randomly initialize model parameters.",
    )
    parser.add_argument(
        "--pretokenized",
        action="store_true",
        help="Whether input sentences for evaluating surprisals are pertokenized or not.",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set random seed
    RANDOM_SEED = (
        args.seed if args.seed is not None else int(np.random.random() * 10000)
    )
    enable_determinism(RANDOM_SEED)
    print("Random seed: {}".format(RANDOM_SEED), file=sys.stderr)

    # Data loading not working!
    # todo: this needs to be done using a correct dataloader, but I need to have test files for this
    # load action ngram list and initialize embeddings
    with open("bllip-lg_action_ngram_list.txt") as f:
        lines = f.readlines()
    symbols = ["<pad>", "_"] + [line.strip().split()[0] for line in lines]

    # Initialize the model
    sclm_config = ScLMConfig(
        is_random_init=args.random_init,
        action_ngram_list=symbols,
        model_name="gpt2",
        cache_dir="pretrained/gpt2",
    )
    sclm = ScLM(sclm_config).to(device)

    w_boundary_char = sclm.w_boundary_char

    print("Scaffold type: {}".format(args.scaffold_type), file=sys.stderr)
    print(
        "Interpolation weight of structure prediction loss {}".format(args.alpha),
        file=sys.stderr,
    )

    # Train
    if args.do_train:
        # Path to save the newly trained model
        MODEL_PATH = (
            args.model_path
            if args.model_path is not None
            else "sclm-{}_pid{}.params".format(args.scaffold_type, os.getpid())
        )
        # print out training settings
        print("Training batch size: {}".format(args.batch_size), file=sys.stderr)
        print("Learning rate: {}".format(args.lr), file=sys.stderr)
        print("Model path: {}".format(MODEL_PATH), file=sys.stderr)

        # Load train and dev data
        train_data_path = args.train_data
        dev_data_path = args.dev_data
        print("Loading train data from {}".format(train_data_path), file=sys.stderr)
        train_lines = load_data(
            train_data_path, sclm.tokenizer, BOS_token=sclm.tokenizer.bos_token
        )
        print("Loading dev data from {}".format(dev_data_path), file=sys.stderr)
        dev_lines = load_data(
            dev_data_path, sclm.tokenizer, BOS_token=sclm.tokenizer.bos_token
        )

        training_args = TrainingArguments(
            output_dir=MODEL_PATH,
            logging_dir=MODEL_PATH,
            per_device_train_batch_size=args.batch_size,
            # gradient_checkpointing=True,
            do_train=True,
            lr_scheduler_type="constant",
            num_train_epochs=args.epochs,
            evaluation_strategy="steps"
            if ((args.valid_every is None) or (args.valid_every < 1))
            else "no",
            eval_steps=args.valid_every,
            logging_strategy="steps",
            loggging_steps=args.report,
            save_strategy="epoch",
        )

        trainer = TrainerWithEvalLoss(  # or Trainer
            model=model,
            args=training_args,
            train_dataset=train_lines,  # todo: replace by correct files if needed!
            eval_dataset=dev_lines,  # todo: replace by correct files if needed!
            tokenizer=sclm.tokenizer,
            optimizers=(AdamW(sclm.parameters(), lr=args.lr), None),
        )

        if args.restore_from is not None:
            trainer.train(resume_from_checkpoint=args.restore_from)
        else:
            trainer.train()

        # todo: need to add regular sampling and early stopping using EarlyStoppingCallback
        # early_stopping_counter = utils.EarlyStopping(best_validation_loss=best_validation_loss, no_improvement_count=no_improvement_count, threshold=args.early_stopping_threshold)

    if False:  # todo: update - could possibly work as is, but will need to fit the lib better
        # redefine compute_metrics() ?
        if args.do_test:
            sclm.eval()
            if args.test_data is None:
                raise ValueError("Test data not specified")

            test_data_path = args.test_data

            test_sents = []
            lines = load_sents(args.test_data)
            for line in lines:
                tokens = line.split()
                words = [token for token in tokens if not is_nonterminal(token)]
                test_sents.append(" ".join(words))

            with torch.no_grad():
                ppl = sclm.get_word_ppl(test_sents)
            print("PPL: {}".format(ppl))

        # Estimate token surprisal values for unparsed sentences
        if args.do_eval:
            sclm.eval()

            if args.fpath is not None:
                sents = load_sents(args.fpath)
            else:
                sents = [
                    "The dogs under the tree are barking.",
                    "The dogs under the tree is barking.",
                    "The keys to the cabinet are on the table.",
                    "The keys to the cabinet is on the table.",
                ]

            print("sentence_id\ttoken_id\ttoken\tsurprisal")

            for i, sent in enumerate(sents):
                if args.pretokenized:
                    words = sent.strip().split()
                    stimulus = sent.strip()
                else:
                    words = nltk.word_tokenize(sent.strip())
                    stimulus = " ".join(words)

                tokens = sclm.tokenizer.tokenize(stimulus)
                with torch.no_grad():
                    surprisals = sclm.get_surprisals(tokens, add_bos_token=True)

                index = 0
                for j, word in enumerate(words):
                    w_str = ""
                    w_surprisal = 0
                    while index < len(tokens) and w_str != word:
                        token_str = tokens[index]
                        if token_str.startswith(w_boundary_char):
                            w_str += token_str[1:]
                        else:
                            w_str += token_str
                        w_surprisal += surprisals[index]

                        index += 1

                    print("{}\t{}\t{}\t{}".format(i + 1, j + 1, word, w_surprisal))
