import logging
import sys
import argparse
import os
import json
import transformers

transformers.logging.set_verbosity_info()

from transformers import TrainerCallback
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from datasets import load_dataset

from sagemaker_tune.report import Reporter
from sagemaker_tune.constants import SMT_CHECKPOINT_DIR


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--eval_steps", type=int, default=32)
    parser.add_argument("--output_data_dir", type=str, default='./output')

    parser.add_argument(f"--{SMT_CHECKPOINT_DIR}", type=str)

    args, _ = parser.parse_known_args()

    params = vars(args)

    checkpoint_dir = params[SMT_CHECKPOINT_DIR]

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # load datasets

    train_dataset, test_dataset = load_dataset(
        'imdb', split=['train', 'test'], cache_dir='./output')


    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True)


    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    logger.info(f" loaded train_dataset length is: {len(train_dataset)}")
    logger.info(f" loaded test_dataset length is: {len(test_dataset)}")

    # compute metrics function for binary classification
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)

    epochs = args.epochs

    report = Reporter()

    # define training args
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        evaluation_strategy="steps",
        logging_dir=f"{args.output_data_dir}/logs",
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        save_strategy='steps',
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        save_total_limit=1,
    )

    # create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )

    # add a callback so that accuracy is sent to Sagemaker Tune whenever it is computed
    class Callback(TrainerCallback):
        def __init__(self):
            self.iteration = 1

        def on_evaluate(self, args, state, control, metrics, **kwargs):
            # Feed the validation accuracy back to Tune
            report(iteration=self.iteration, accuracy=metrics['eval_accuracy'])
            self.iteration += 1


    trainer.add_callback(Callback())

    if os.listdir(checkpoint_dir) == []:
        trainer.train()
    else:
        # train model
        trainer.train(resume_from_checkpoint=os.path.join(checkpoint_dir, os.listdir(checkpoint_dir)[0]))

    # evaluate model
    eval_result = trainer.evaluate(eval_dataset=test_dataset)

    # writes eval result to file which can be accessed later in s3 ouput
    with open(os.path.join(args.output_data_dir, "eval_results.txt"), "w") as writer:
        print(f"***** Eval results *****")
        for key, value in sorted(eval_result.items()):
            writer.write(f"{key} = {value}\n")

