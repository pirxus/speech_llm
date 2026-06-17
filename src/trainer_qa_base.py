import argparse
import sys
import os
import torch
import json

from utils import create_logger, get_batch, TestSetWrapper, load_es_commonvoice, create_scheduler
from qa.qa_dataset import SpanishQADataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoTokenizer
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from accelerate import Accelerator
from speech_llm_model import SpeechLLMBase
from collators import create_collator

from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs

def main(args):

    # Initialize accelerator
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=2000))
    accelerator = Accelerator(mixed_precision='bf16', kwargs_handlers=[kwargs])

    if accelerator.is_local_main_process:
        # create the destination directory
        os.makedirs(args.out_dir, exist_ok=True)

        # set up logger
        logger = create_logger(os.path.join(args.out_dir, "logs/train"), verbose=True)
        logger.info(f"Process rank: {accelerator.process_index}")
        logger.info(f"Number of processes: {accelerator.num_processes}")

    # load the model
    if args.from_pretrained:
        if accelerator.is_local_main_process:
            logger.info("Loading pretrained model..")

        model, model_args = SpeechLLMBase.from_pretrained(
            args.from_pretrained,
            device=accelerator.device,
            return_model_args=True,
        )

        # if the warm checkpoint has no LoRA but we want to add adapters now, do so
        if args.lora_r is not None and model_args.get('lora_r') is None:
            if accelerator.is_local_main_process:
                logger.info(f"Adding LoRA adapters (r={args.lora_r}, alpha={args.lora_a or args.lora_r}) to pretrained model")
            model.add_lora_adapters(args.lora_r, args.lora_a)
            model_args['lora_r'] = args.lora_r
            model_args['lora_a'] = args.lora_a

    else:
        # create the model config
        model_args = dict(
            speech_enc_id=args.speech_enc_id,
            llm_id=args.llm_id,
            lora_r=args.lora_r,
            lora_a=args.lora_a,
            connector_config=dict(
                num_layers=args.num_layers,
                attn_heads=args.num_heads,
                hidden_size=args.hidden_size,
                intermediate_size=args.intermediate_size,
                downsampling_factor=args.downsampling_factor,
                norm_first=args.norm_first,
                add_final_norm=args.add_final_norm,
                use_positional_embeddings=args.use_positional_embeddings,
                dropout=args.dropout,
            ),
        )

        # instantiate the model from config
        model = SpeechLLMBase(**model_args)

    if accelerator.is_local_main_process:
        # save the model args to the experiment directory
        with open(os.path.join(args.out_dir, "config.json"), "w") as f:
            json.dump(model_args, f, indent=4)

        logger.info("\nFinished loading model..\n")
        logger.info(model)

        # print the model parameters
        params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {params:,}, Trainable parameters: {trainable_params:,}")


    # prepare the model
    model = accelerator.prepare(model)

    # create the speech preprocessor for dataset and load the tokenizer
    speech_processor = AutoProcessor.from_pretrained(args.speech_enc_id)
    tokenizer = AutoTokenizer.from_pretrained(args.llm_id)

    # Create training dataset, batch sampler and a dataloader

    RETRIEVAL_JSON = '/mnt/matylda6/isedlacek/projects/eloquence/t2.5/speech_llm/exp/data/m3_text_retrieval.json'
    REPHRASED_EXTRACTIVE = '/mnt/matylda6/isedlacek/projects/eloquence/t2.5/speech_llm/exp/data/rephrased_answers.json'
    print("Use impossible: ", not args.no_use_impossible)
    dataset = {
        args.train_split: SpanishQADataset(
            split='train',
            topk=args.topk,
            use_impossible=not args.no_use_impossible,
            retrieval_json=RETRIEVAL_JSON,
            rephrased_extractive=REPHRASED_EXTRACTIVE
        ),
        args.validation_split: SpanishQADataset(
            split='dev',
            topk=args.topk,
            use_impossible=not args.no_use_impossible,
            retrieval_json=RETRIEVAL_JSON,
            rephrased_extractive=REPHRASED_EXTRACTIVE
        ),
        **{ test_split: SpanishQADataset(
                split=test_split,
                topk=args.topk,
                use_impossible=not args.no_use_impossible,
                retrieval_json=RETRIEVAL_JSON,
                rephrased_extractive=REPHRASED_EXTRACTIVE
            ) for test_split in args.test_splits
        }
    }

    collator = create_collator(
        'SalamandraQACollator', # TODO: make parametrizable
        feature_extractor=speech_processor.feature_extractor, # TODO: collator arguments should have a separate namespace in the config
        tokenizer=tokenizer,
    )

    if args.do_train:

        train_dset = dataset[args.train_split]
        train_loader = DataLoader(
            train_dset,
            batch_size=args.bsize,
            collate_fn=collator,
            shuffle=args.shuffle,
            num_workers=args.nj,
            prefetch_factor=args.nj if args.nj > 0 else None,
        )

        val_dset = dataset[args.validation_split]
        val_loader = DataLoader(
            val_dset,
            batch_size=args.bsize,
            collate_fn=collator,
            num_workers=args.nj,
            prefetch_factor=args.nj if args.nj > 0 else None,
        )

        # create optimizer
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay)

        # create scheduler
        scheduler = create_scheduler(opt, args)

        # Prepare everyting else (other than the model) with the accelerator
        train_loader, val_loader, opt, scheduler = accelerator.prepare(
            train_loader, val_loader, opt, scheduler
        )

        if accelerator.is_local_main_process:
            logger.info(f"Starting training for {args.steps} steps with {accelerator.num_processes} GPUs")
            logger.info(f"Training with batch size {args.bsize} and gradient accumulation steps {args.gradient_accumulation_steps}")
            logger.info(f"Total batch size: {args.bsize * accelerator.num_processes * args.gradient_accumulation_steps}")
            best_checkpoints = []

        # Create infinite data loaders for step-based training
        train_iter = iter(train_loader)

        # Train model based on steps instead of epochs
        global_step = 0
        best_eval_loss = float('inf')
        early_stop_counter = 0

        # put the model into training mode
        model.train()

        # create progress bar and start training
        progress_bar = tqdm(range(args.steps), desc="Training", disable=not accelerator.is_local_main_process)
        for step in progress_bar:

            # Get batch and handle end of iterator
            batch_dict = get_batch(train_iter, train_loader, accelerator, logger if accelerator.is_local_main_process else None)

            """
            if step % 50 == 0:
                generation_config = {
                    "bos_token_id": 1,
                    "eos_token_id": [
                        2,
                        5
                    ],
                    "max_new_tokens": 300,
                    #"do_sample": true,
                    #"temperature": 0.6,
                    #"repetition_penalty": 1.2,
                }
                model.eval()
                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                    speech_out = model.generate(**batch_dict, **generation_config)
                texts = tokenizer.batch_decode(speech_out, skip_special_tokens=True)
                labels = torch.clone(batch_dict['labels'])
                labels[labels == -100] = 2
                labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                print(texts)
                print(labels)
                model.train()
            """

            # forward pass
            model_out = model(**batch_dict)

            # backward pass
            loss = model_out['loss']
            accelerator.backward(loss / args.gradient_accumulation_steps)

            if (step + 1) % args.gradient_accumulation_steps == 0:

                #if args.unfreeze_after_steps > 0 and global_step < args.unfreeze_after_steps:
                #    # zero the gradients of the speech encoder
                #    unwrapped = accelerator.unwrap_model(model)
                #    for i in range(1, args.enc_n_layers_to_unfreeze + 1):
                #        for param in unwrapped.speech_encoder.layers[-i].parameters():
                #            param.grad = None

                opt.step()
                if scheduler:
                    scheduler.step()
                opt.zero_grad()

            # Log loss
            gathered_loss = accelerator.gather(loss.detach()).mean()

            if accelerator.is_local_main_process and global_step % args.logging_steps == 0:
                with logging_redirect_tqdm():
                    logger.info(f"Step: {global_step}, Loss: {gathered_loss.item():.7f}")

                progress_bar.set_postfix({
                    "lr": opt.param_groups[0]["lr"],
                })

            global_step += 1

            # Run evaluation at specified intervals - only on main process
            if global_step % args.eval_steps == 0:
                model.eval()

                if accelerator.is_local_main_process:
                    with logging_redirect_tqdm():
                        logger.info(f"Running evaluation at step {global_step}...")

                with torch.no_grad():
                    eval_losses = []

                    for eval_step, batch_dict in enumerate(tqdm(val_loader, desc="Evaluating", total=len(val_dset) // (args.bsize * accelerator.num_processes), disable=not accelerator.is_local_main_process)):
                        if args.limit_eval_steps and eval_step >= args.limit_eval_steps:
                            break

                        # Use BF16 for evaluation to match training precision
                        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                            model_out = model(**batch_dict)

                        loss = model_out['loss']

                        gathered_loss = accelerator.gather(loss).mean()

                        eval_losses.append(gathered_loss)

                    eval_loss = sum(eval_losses) / len(eval_losses)

                    # log the loss
                    if accelerator.is_local_main_process:
                        with logging_redirect_tqdm():
                            logger.info(f"Step: {global_step}, Eval Loss: {eval_loss.item():.7f}")

                        unwrapped_model = accelerator.unwrap_model(model)

                        # Save checkpoint
                        ckpt_path = os.path.join(args.out_dir, f"model_step_{global_step}.pt")
                        torch.save(unwrapped_model.state_dict(), ckpt_path)

                        # Maintain top-n best checkpoints
                        best_checkpoints.append((eval_loss, ckpt_path))
                        best_checkpoints.sort()  # sort by loss (lowest first)

                        if len(best_checkpoints) > args.n_best:
                            _, worst_path = best_checkpoints.pop(-1)
                            if os.path.exists(worst_path):
                                os.remove(worst_path)
                                with logging_redirect_tqdm():
                                    logger.info(f"Removed checkpoint: {worst_path}")

                        # Save best model
                        if eval_loss < best_eval_loss:
                            best_eval_loss = eval_loss
                            early_stop_counter = 0
                            torch.save(unwrapped_model.state_dict(), os.path.join(args.out_dir, "model_best.pt"))
                            with logging_redirect_tqdm():
                                logger.info(f"New best model saved with eval loss: {eval_loss}")
                        else:
                            early_stop_counter += 1
                            with logging_redirect_tqdm():
                                logger.info(f"No improvement for {early_stop_counter} evaluations")

                        del unwrapped_model

                # Sync early stopping decision across processes
                should_stop = torch.tensor(1.0 if early_stop_counter >= args.early_stopping_patience else 0.0,
                                          device=accelerator.device)
                should_stop = accelerator.reduce(should_stop, reduction="sum")
                if should_stop.item() > 0:
                    if accelerator.is_local_main_process:
                        with logging_redirect_tqdm():
                            logger.info(f"Early stopping after {global_step} steps")
                    break

                # Return to training mode for all processes
                model.train()

            # Stop training after specified steps
            if global_step >= args.steps:
                break

        if accelerator.is_local_main_process:
            logger.info(f"Training completed after {global_step} steps")

    if args.do_generate:

        # unwrap the model -- we are going to be using generate()
        unwrapped = accelerator.unwrap_model(model)

        generation_config = {
            "bos_token_id": 1,
            "pad_token_id": 2,
            "eos_token_id": [
            2,
            5
            ],
            #"do_sample": true,
            #"temperature": 0.6,
            #"repetition_penalty": 1.2,
            "max_new_tokens": 300,
        }

        if accelerator.is_local_main_process:
            global_test_predictions = {}

        for test_split in args.test_splits:
            predictions = []
            gts = []
            item_ids = []
            uuids = []

            test_dset = TestSetWrapper(dataset[test_split])
            test_loader = DataLoader(
                test_dset,
                batch_size=args.bsize,
                collate_fn=collator,
                num_workers=args.nj,
                prefetch_factor=args.nj if args.nj > 0 else None,
            )

            # prepare the test loader
            test_loader = accelerator.prepare(test_loader)

            for batch_dict in tqdm(test_loader, desc=f"Running prediction on test split '{test_split}'", total=len(test_dset) // (args.bsize * accelerator.num_processes), disable=not accelerator.is_local_main_process):

                # get the item ids for de-duplication afterwards
                item_indices = batch_dict.pop('item_indices')
                labels_text = batch_dict.pop('labels_text')
                item_uuids = batch_dict.pop('id') # uuids for prediction storage

                with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                    speech_out = unwrapped.generate(**batch_dict, **generation_config)
                    texts = tokenizer.batch_decode(speech_out, skip_special_tokens=True)

                    for item_id, label_text, prediction, uuid in zip(item_indices, labels_text, texts, item_uuids):
                        gts.append(label_text)
                        predictions.append(prediction)
                        item_ids.append(item_id)
                        uuids.append(uuid)

            # sync up and gather
            accelerator.wait_for_everyone()

            if accelerator.num_processes > 1:
                gathered_predictions = accelerator.gather_for_metrics(predictions)
                gathered_gts = accelerator.gather_for_metrics(gts)
                gathered_item_ids = accelerator.gather_for_metrics(item_ids)
                gathered_uuids = accelerator.gather_for_metrics(uuids)

                if accelerator.is_local_main_process:
                    predictions = gathered_predictions
                    gts = gathered_gts
                    item_ids = gathered_item_ids
                    uuids = gathered_uuids

            if accelerator.is_local_main_process:
                # first, deduplicate the predictions, there can be multiple same ids because of batching
                unique_predictions = []
                unique_gts = []
                unique_ids = []
                unique_uuids = []
                for prediction, gt, item_id, uuid in zip(predictions, gts, item_ids, uuids):
                    if item_id not in unique_ids:
                        unique_predictions.append(prediction)
                        unique_gts.append(gt)
                        unique_ids.append(item_id)
                        unique_uuids.append(uuid)

                # let's sort the predictions by item_id
                unique_predictions = [x for _, x in sorted(zip(unique_ids, unique_predictions))]
                unique_gts = [x for _, x in sorted(zip(unique_ids, unique_gts))]
                unique_uuids = [x for _, x in sorted(zip(unique_ids, unique_uuids))]

                global_test_predictions[test_split] = {
                    "predictions": unique_predictions,
                    "labels": unique_gts,
                    "uuids": unique_uuids,
                }

        # write the predictions to a file
        if accelerator.is_local_main_process:
            save_path = os.path.join(args.out_dir, "test_predictions.json")
            logger.info(f"Saving predictions to {save_path}")

            with open(save_path, "w") as f:
                json.dump(global_test_predictions, f, indent=4)


def parse_args():
    """parse command line arguments"""

    parser = argparse.ArgumentParser()

    io_group = parser.add_argument_group("Input and output related")
    io_group.add_argument(
        "--out_dir", type=str, required=True, help="path to the output directory to save experiment results, logs and checkpoints"
    )

    model_group = parser.add_argument_group("Model definition related group")
    model_group.add_argument("--speech_enc_id", default="openai/whisper-small.en")
    model_group.add_argument("--llm_id", default="meta-llama/Llama-3.1-8B-Instruct")
    model_group.add_argument("--from_pretrained", type=str, help="path to the pretrained joint model")
    model_group.add_argument("--lora_r", type=int, default=None, help="LoRA rank; if not set, no LoRA adapters are added")
    model_group.add_argument("--lora_a", type=int, default=None, help="LoRA alpha; defaults to lora_r if not set")

    opt_group = parser.add_argument_group("Training and Optimization related")
    #opt_group.add_argument("--ngpus", type=int, default=1, help="number of gpus")
    opt_group.add_argument("--lr", type=float, default=1e-4, help="learning rate")

    opt_group.add_argument("--wdecay", type=float, default=0.0, help="weight decay")
    opt_group.add_argument("--steps", type=int, default=10000, help="number of training epochs")
    opt_group.add_argument("--eval_steps", type=int, default=100, help="number of training epochs")
    opt_group.add_argument("--limit_eval_steps", type=int, default=None, help="limit the number of evaluation steps")
    opt_group.add_argument("--logging_steps", type=int, default=10, help="logging steps")
    opt_group.add_argument("--warmup", type=int, default=1, help="number of warmup steps")
    opt_group.add_argument("--scheduler", type=str, default='linear', help="Scheduler type to use. Choose between 'cosine' and 'linear'")
    opt_group.add_argument("--gradient_accumulation_steps", type=int, default=1, help="gradient_accumulation_steps")
    opt_group.add_argument("--n_best", type=int, default=3, help="number of best checkpoints to keep")
    opt_group.add_argument("--early_stopping_patience", type=int, default=3, help="number of evaluations without improvement before early stopping")

    opt_group.add_argument("--nj", type=int, default=0, help="number of workers")
    opt_group.add_argument("--shuffle", action="store_true", help="shuffle the data")
    opt_group.add_argument("--bsize", type=int, default=128, help="batch size")
    opt_group.add_argument("--seed", type=int, default=None, help="seed for rng init")

    connector_group = parser.add_argument_group("Connector related arguments")
    connector_group.add_argument("--downsampling_factor", type=int, default=6, help="downsampling factor before the connnector")
    connector_group.add_argument("--num_heads", type=int, default=16, help="number of connector attention heads")
    connector_group.add_argument("--num_layers", type=int, default=2, help="number of connector layers")
    connector_group.add_argument("--hidden_size", type=int, default=1024, help="number of layers for the connector")
    connector_group.add_argument("--intermediate_size", type=int, default=4096, help="connector intermediate projection size")
    connector_group.add_argument("--norm_first", action="store_true", help="Whether to use pre-norm in the connector")
    connector_group.add_argument("--add_final_norm", action="store_true", help="Whether add one last layernorm before the LM projection at the end of the connector")
    connector_group.add_argument("--use_positional_embeddings", action="store_true", help="Whether to use sinusoidal positional embeddings in the connector")
    connector_group.add_argument("--dropout", type=float, default=0.1, help="dropout factor for the connector")

    data_group = parser.add_argument_group("Data related arguments")
    data_group.add_argument("--train_split", type=str, default="train", help="Train split name")
    data_group.add_argument("--validation_split", type=str, default="validation", help="Validation split name")
    data_group.add_argument("--test_splits", type=str, default="test", nargs="+", help="List of test split names")
    data_group.add_argument("--topk", type=int, default=5, help="Max topk retrieved chunks used in the model input")
    data_group.add_argument("--no_use_impossible", action="store_true", help="Whether to use the impossible questions or not")

    control_group = parser.add_argument_group("Trainer control related arguments")
    control_group.add_argument("--do_train", action="store_true", help="Runs training in the trainer script")
    control_group.add_argument("--do_generate", action="store_true", help="Runs test split prediction in the trainer script")


    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    if args.seed is not None: torch.manual_seed(args.seed)

    main(args)
