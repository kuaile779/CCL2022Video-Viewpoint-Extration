import argparse
from transformers import AutoModel, AutoConfig,  AutoTokenizer
from model import cvModel
from data_process import *
from eval_utils import compute_predictions, compute_predictions_in_dev
import torch
import random
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
import logging
import torch.nn as nn
from tqdm import trange, tqdm
import timeit
import os
from os.path import join
from transformers import logging as lg
lg.set_verbosity_error()


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def train(args, train_dataset: TensorDataset, model: nn.Module,
          tokenizer):
    args.train_batch_size = args.batch_size
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    logging.info("***** Running training *****")
    logging.info("  Total training samples number = %d", len(train_dataset))
    logging.info("  Num Epochs = %d", args.num_train_epochs)
    logging.info("  Instantaneous batch size per GPU = %d", args.batch_size)
    logging.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps,
        )
    logging.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logging.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch",
    )
    set_seed(args)

    best_f1 = 0.0
    index = 0
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_labels": batch[3],
                "end_labels": batch[4],
            }
            outputs = model(**inputs)
            loss = outputs
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            tr_loss += loss.item()
            logging_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging.info(f"Current global step: {global_step}")
                    logging.info(f"average loss of batch: {logging_loss / args.logging_steps}")
                    logging_loss = 0

                if args.saving_steps > 0 and global_step % args.saving_steps == 0:
                    logging.info(f"Current global step: {global_step}, start evaluating!")
                    temp_f1 = evaluate(args, model, global_step, prefix=f"{global_step}-valid",
                                       eval_data_dir=args.eval_data_dir)
                    if temp_f1 > best_f1:
                        best_f1 = temp_f1
                        index += 1
                        logging.info(f"Start evaluating in dev!")
                        dev_f1 = evaluate_in_dev(args, model, global_step, index, prefix=f"{global_step}-dev",
                                           eval_data_dir=args.eval_data_dir)
                        logging.info(f"End evaluating in dev!")
                    logging.info(f"Evaluation result in valid-set at global step {global_step}")
                    logging.info(f"------------Current best f1: {best_f1}-----------")

    return global_step, tr_loss / global_step


def evaluate(args, model, global_step, prefix, eval_data_dir: str):
    logging.info(f"Loading data for evaluation from {eval_data_dir}!")
    examples = load_file(args.eval_data_dir + "/valid_examples.obj")
    features = load_file(args.eval_data_dir + "/valid_features.obj")
    dataset = convert_features_to_dataset(features, is_training=False)
    logging.info("Complete Loading!")

    args.eval_batch_size = args.batch_size
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    logging.info("***** Running evaluation {} *****".format(prefix))
    logging.info("  Num examples = %d", len(dataset))
    logging.info("  Batch size = %d", args.eval_batch_size)

    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=True):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }
            features_indexes = batch[5]
            outputs = model(**inputs)

        for i, feature_index in enumerate(features_indexes):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.pid)
            output = [to_list(o[i]) for o in outputs]

            start_logits, end_logits = output

            result = (unique_id, start_logits, end_logits)
            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logging.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    res = compute_predictions(args, examples, features, all_results, global_step)
    return res

def evaluate_in_dev(args, model, global_step, index, prefix, eval_data_dir: str):
    logging.info(f"Loading data for evaluation from {eval_data_dir}!")
    examples = load_file(args.eval_data_dir + "/test_examples.obj")
    features = load_file(args.eval_data_dir + "/test_features.obj")
    dataset = convert_features_to_dataset(features, is_training=False)
    logging.info("Complete Loading!")

    args.eval_batch_size = args.batch_size
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    logging.info("***** Running evaluation {} *****".format(prefix))
    logging.info("  Num examples = %d", len(dataset))
    logging.info("  Batch size = %d", args.eval_batch_size)

    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=True):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }
            features_indexes = batch[5]
            outputs = model(**inputs)

        for i, feature_index in enumerate(features_indexes):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.pid)
            output = [to_list(o[i]) for o in outputs]

            start_logits, end_logits = output

            result = (unique_id, start_logits, end_logits)
            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logging.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    res = compute_predictions_in_dev(args, examples, features, all_results, global_step, index)
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default='./model',
        type=str,
        help="Path to Electra model."
    )

    parser.add_argument(
        "--train_data_dir",
        default='./data',
        type=str,
        help="The directory which contain the generated train-set examples and features file."
    )

    parser.add_argument(
        "--eval_data_dir",
        default='./data',
        type=str,
        help="The directory which contain the generated dev-set examples and features file."
    )

    parser.add_argument(
        "--output_dir",
        default='./output',
        type=str,
        help="The directory which contain the generated dev-set examples and features file."
    )

    parser.add_argument("--device", default="cuda", type=str, help="Whether not to use CUDA when available")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU/CPU for training and evaluating.")
    parser.add_argument("--learning_rate", default=7e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization.")
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--num_train_epochs", default=10.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--logging_steps", type=int, default=200, help="Log every X updates steps.")
    parser.add_argument("--saving_steps", type=int, default=400, help="Log every X updates steps.")

    parser.add_argument(
        "--multi_span_threshold",
        type=float,
        default=0.6,
        help="Span which score is bigger than (max_span_score * multi_span_threshold) will also be output!"
    )

    args = parser.parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logging.info("All input parameters:")
    print(json.dumps(vars(args), sort_keys=False, indent=2))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    set_seed(args)

    config = AutoConfig.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    pretrain_model = AutoModel.from_pretrained(args.model_path, config=config)
    model = cvModel(pretrain_model)
    model.to(device=args.device)

    logging.info("Loading pre-processed examples and features!")
    train_examples = load_file(args.train_data_dir + "/train_examples.obj")
    train_features = load_file(args.train_data_dir + "/train_features.obj")
    logging.info("Complete loading!")

    logging.info("Converting features to pytorch Dataset!")
    train_dataset = convert_features_to_dataset(train_features, is_training=True)
    logging.info("Complete converting!")

    global_step, tr_loss = train(args, train_dataset, model, tokenizer)
    logging.info("global_step = %s, average loss = %s", global_step, tr_loss)
    logging.info("Training Complete!")


if __name__ == "__main__":
    main()



