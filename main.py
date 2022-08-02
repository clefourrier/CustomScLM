from modeling_sclm import ScLM
from torch.optim import AdamW
from transformers import TrainingArguments, Trainer
#from configuration_sclm import ScLMConfig # todo, useless for now

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset, Dataset



def is_nonterminal(token):
    """
    Check if a token is a nonterminal action or word piece token.
    """
    if (token.startswith('NT(') and token.endswith(')')) or token == 'REDUCE()':
        return True
    else:
        return False


def load_sents(path):
    # todo: save dataset on hugging face to load it here? or use as preprocessing step
    with open(path) as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line != '']
    return lines


def load_data(path, tokenizer, BOS_token=None):
    # todo: add dataloader here, with tokenizer map
    with open(path) as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line != '']

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
                    action_ngrams.append('_'.join(action_seq))
                    action_seq = []
                else:
                    action_ngrams.append('_')
                words.append(token)

        action_ngrams.append('_'.join(action_seq)) # add the action ngram that comes after the last word

        sent = ' '.join(words)
        word_pieces = tokenizer.tokenize(sent)

        combined = ''
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
                action_ngram_seq += [action_ngrams[word_index]] + ['_' for _ in range(n_piece-1)]
                combined = ''
                n_piece = 0
                word_index += 1
        assert combined == ''
        assert word_index == len(words)

        action_ngram_seq.append(action_ngrams[-1])

        assert len(word_pieces) == (len(action_ngram_seq) - 1)

        if BOS_token is None:
            data.append([word_pieces, action_ngram_seq])
        else:
            data.append([[BOS_token] + word_pieces, action_ngram_seq])

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, help='Path to training data.')
    parser.add_argument('--dev_data', type=str, help='Path to validation data.')
    parser.add_argument('--test_data', type=str, help='Path to test data.')
    parser.add_argument('--fpath', type=str, help='File path for estimating surprisals.')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate.')
    parser.add_argument('--alpha', type=float, default=0.5, help='Hyerparameter in (0, 1) for weighting the structure prediction loss against the word prediction loss. Default is 0.5.')
    parser.add_argument('--scaffold_type', type=str, help='Type of scaffold. (next, past)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--report', type=int, default=1000, help='Frequency of report training status after number of training batches.')
    parser.add_argument('--valid_every', type=int, default=None, help='Frequency of validating and saving model parameters after number of training batches.')
    parser.add_argument('--sample_every', type=int, default=10000, help='Frequency of generating samples from the model during training.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed.')
    parser.add_argument('--do_train', action='store_true', help='Whether to train the model.')
    parser.add_argument('--do_test', action='store_true', help='Whether to test the model.')
    parser.add_argument('--do_eval', action='store_true', help='Whether to use the model for surprisal estimation.')
    parser.add_argument('--model_path', type=str, default=None, help='Path of the model to be trained and saved.')
    parser.add_argument('--restore_from', type=str, default=None, help='Path to the trained model checkpoint. Will use the pretrained model if path not specified.')
    parser.add_argument('--batch_size', type=int, default=5, help="Size of a training batch.")
    parser.add_argument('--early_stopping_threshold', type=int, default=2, help='Threshold for early stopping.')
    parser.add_argument('--random_init', action='store_true', help="Randomly initialize model parameters.")
    parser.add_argument('--pretokenized', action='store_true', help="Whether input sentences for evaluating surprisals are pertokenized or not.")

    args = parser.parse_args()

    log_softmax = torch.nn.LogSoftmax(-1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set random seed
    RANDOM_SEED = args.seed if args.seed is not None else int(np.random.random()*10000)
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    print('Random seed: {}'.format(RANDOM_SEED), file=sys.stderr)

    # load action ngram list and initialize embeddings
    with open('bllip-lg_action_ngram_list.txt') as f:
        lines = f.readlines()
    symbols = ['<pad>', '_'] + [line.strip().split()[0] for line in lines]

    # Initialize the model
    sclm = ScLM(is_random_init=args.random_init, action_ngram_list=symbols, device=device, model_name='gpt2')
    w_boundary_char = sclm.w_boundary_char

    # Load model checkpoint
    if args.restore_from is not None:
        print('Load parameters from {}'.format(args.restore_from), file=sys.stderr)
        checkpoint = torch.load(args.restore_from)
        sclm.model.load_state_dict(checkpoint['model_state_dict'])

    SCAFFOLD_TYPE = args.scaffold_type
    print('Scaffold type: {}'.format(SCAFFOLD_TYPE), file=sys.stderr)
    ALPHA = args.alpha
    print('Interpolation weight of structure prediction loss {}'.format(ALPHA), file=sys.stderr)

    # Train
    if args.do_train:
        # Path to save the newly trained model
        MODEL_PATH = args.model_path if args.model_path is not None else "sclm-{}_pid{}.params".format(SCAFFOLD_TYPE, os.getpid())
        # print out training settings
        print('Training batch size: {}'.format(args.batch_size), file=sys.stderr)
        print('Learning rate: {}'.format(args.lr), file=sys.stderr)
        print('Model path: {}'.format(MODEL_PATH), file=sys.stderr)

        # Load train and dev data
        train_data_path = args.train_data
        dev_data_path = args.dev_data
        print("Loading train data from {}".format(train_data_path), file=sys.stderr)
        train_lines = load_data(train_data_path, sclm.tokenizer, BOS_token=sclm.tokenizer.bos_token)
        print("Loading dev data from {}".format(dev_data_path), file=sys.stderr)
        dev_lines = load_data(dev_data_path, sclm.tokenizer, BOS_token=sclm.tokenizer.bos_token)

        if args.restore_from is not None:
            sclm.eval()
            with torch.no_grad():
                validation_loss, _, _ = sclm.get_validation_loss(dev_lines, scaffold_type=SCAFFOLD_TYPE)
            best_validation_loss = validation_loss
            sclm.train()
            print('resume training; validation loss: {}'.format(best_validation_loss))
        else:
            best_validation_loss = np.inf

        training_args = TrainingArguments(
            output_dir=MODEL_PATH,
            logging_dir=f"{output_dir}/{model_name}/{data_seed}",
            per_device_train_batch_size=10,
            gradient_accumulation_steps=10,
            gradient_checkpointing=True,
            do_train=True,
            lr_scheduler_type="constant",
            num_train_epochs=args.epochs,
            evaluation_strategy="steps" if ((args.valid_every is None) or (args.valid_every < 1)) else "no",
        	eval_steps=args.valid_every,
            logging_strategy="steps",
            loggging_steps=args.report,
            save_strategy="epoch",
        )
        print(training_args.device)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset_tk_train,
            eval_dataset=dataset_tk_test,
            compute_metrics=compute_metrics_pretraining,
            tokenizer=tokenizer,
            optimizers=(AdamW(sclm.parameters(), lr=args.lr), None),
        )

        starting_epoch = checkpoint['epoch'] + 1 if (args.restore_from is not None) else 0
        no_improvement_count = checkpoint['no_improvement_count'] if (args.restore_from is not None) else 0


        trainer.train()
        # need to add regular sampling, early stopping using EarlyStoppingCallback 
        #early_stopping_counter = utils.EarlyStopping(best_validation_loss=best_validation_loss, no_improvement_count=no_improvement_count, threshold=args.early_stopping_threshold)

    if args.do_test:
        sclm.eval()
        if args.test_data is None:
            raise ValueError('Test data not specified')

        test_data_path = args.test_data

        test_sents = []
        lines = load_sents(args.test_data)
        for line in lines:
            tokens = line.split()
            words = [token for token in tokens if not is_nonterminal(token)]
            test_sents.append(' '.join(words))

        with torch.no_grad():
            ppl = sclm.get_word_ppl(test_sents)
        print('PPL: {}'.format(ppl))

    # Estimate token surprisal values for unparsed sentences
    if args.do_eval:
        sclm.eval()

        if args.fpath is not None:
            sents = load_sents(args.fpath)
        else:
            sents = ["The dogs under the tree are barking.", "The dogs under the tree is barking.",
                    "The keys to the cabinet are on the table.", "The keys to the cabinet is on the table.",]

        print('sentence_id\ttoken_id\ttoken\tsurprisal')

        for i, sent in enumerate(sents):
            if args.pretokenized:
                words = sent.strip().split()
                stimulus = sent.strip()
            else:
                words = nltk.word_tokenize(sent.strip())
                stimulus = ' '.join(words)

            tokens = sclm.tokenizer.tokenize(stimulus)
            with torch.no_grad():
                surprisals = sclm.get_surprisals(tokens, add_bos_token=True)

            index = 0
            for j, word in enumerate(words):
                w_str = ''
                w_surprisal = 0
                while index < len(tokens) and w_str != word:
                    token_str = tokens[index]
                    if token_str.startswith(w_boundary_char):
                        w_str += token_str[1:]
                    else:
                        w_str += token_str
                    w_surprisal += surprisals[index]

                    index += 1

                print('{}\t{}\t{}\t{}'.format(i+1, j+1, word, w_surprisal))
