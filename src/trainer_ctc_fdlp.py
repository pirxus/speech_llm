import argparse
import os
import torch
import torch.nn as nn
import sys
import json

from datasets import load_from_disk
from utils import create_logger, get_batch, TestSetWrapper
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from accelerate import Accelerator
from speech_llm_model import CTCSpeechEncoder
from collators import create_collator

from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs


def greedy_ctc_decode(logits, vocab_dict, blank_id=0):
    """
    Greedy CTC decoding: take argmax at each timestep and collapse repeats/blanks.
    
    Args:
        logits: (batch, time, vocab_size) tensor of logits
        vocab_dict: dictionary mapping characters to indices (e.g., {"a": 2, "b": 3})
        blank_id: ID of the blank token (default 0)
        
    Returns:
        List of decoded strings
    """
    # Create inverse vocabulary (index -> character)
    inv_vocab = {v: k for k, v in vocab_dict.items()}
    
    # Get predictions (batch, time)
    preds = torch.argmax(logits, dim=-1)
    
    decoded_texts = []
    for pred in preds:
        # Convert to list and collapse
        pred_list = pred.tolist()
        
        # Collapse repeats and remove blanks
        collapsed = []
        prev_id = None
        for token_id in pred_list:
            if token_id != blank_id and token_id != prev_id:
                collapsed.append(token_id)
            prev_id = token_id
        
        # Convert to text
        text = "".join([inv_vocab.get(tid, "") for tid in collapsed])
        decoded_texts.append(text)
    
    return decoded_texts


def main(args):
    # Initialize accelerator
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=2000))
    accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=[kwargs])

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

        model, model_args = CTCSpeechEncoder.from_pretrained(
            args.from_pretrained,
            device=accelerator.device,
            return_model_args=True,
        )

    else:
        # Create FDLP config
        fdlp_config = {
            "n_filters": args.n_filters,
            "coeff_num": args.coeff_num,
            "coeff_range": args.coeff_range,
            "order": args.order,
            "fduration": args.fduration,
            "frate": args.frate,
            "overlap_fraction": args.overlap_fraction,
            "srate": 16000,
        }

        # create the collator first to get vocab size
        collator = create_collator(
            "FDLPCTCCollator",
            fdlp_config=fdlp_config,
            label_column=args.label_column,
        )

        # create the model config
        model_args = dict(
            vocab_size=collator.vocab_size,
            d_model=args.d_model,
            num_mel_bins=args.n_filters,  # FDLP features dimension
            num_encoder_layers=args.num_encoder_layers,
            num_attention_heads=args.num_attention_heads,
            intermediate_size=args.intermediate_size,
            dropout=args.dropout,
            max_source_positions=args.max_source_positions,
        )

        # instantiate the model from config
        model = CTCSpeechEncoder(**model_args)

    if accelerator.is_local_main_process:
        # save the model args to the experiment directory
        with open(os.path.join(args.out_dir, "config.json"), "w") as f:
            json.dump(model_args, f, indent=4)

        logger.info("\nFinished loading model..\n")
        logger.info(model)

        # print the model parameters
        params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(
            f"Total parameters: {params:,}, Trainable parameters: {trainable_params:,}"
        )

    # prepare the model
    model = accelerator.prepare(model)

    # Create training dataset, batch sampler and a dataloader
    dataset = load_from_disk(args.data_path)

    # Create collator if not already created
    if not args.from_pretrained:
        # Collator already created above for vocab_size
        pass
    else:
        # Create FDLP config
        fdlp_config = {
            "n_filters": args.n_filters,
            "coeff_num": args.coeff_num,
            "coeff_range": args.coeff_range,
            "order": args.order,
            "fduration": args.fduration,
            "frate": args.frate,
            "overlap_fraction": args.overlap_fraction,
            "srate": 16000,
        }
        
        collator = create_collator(
            "FDLPCTCCollator",
            fdlp_config=fdlp_config,
            label_column=args.label_column,
        )

    if args.do_train:
        train_dset = dataset[args.train_split]
        train_loader = DataLoader(
            train_dset,
            batch_size=args.bsize,
            collate_fn=collator,
            shuffle=True,
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
        opt = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.wdecay
        )

        # warmup scheduler
        if args.warmup:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                opt, lr_lambda=lambda step: min((step + 1) / args.warmup, 1.0)
            )
        else:
            scheduler = None

        # Prepare everyting else (other than the model) with the accelerator
        train_loader, val_loader, opt, scheduler = accelerator.prepare(
            train_loader, val_loader, opt, scheduler
        )

        if accelerator.is_local_main_process:
            logger.info(
                f"Starting training for {args.steps} steps with {accelerator.num_processes} GPUs"
            )
            logger.info(
                f"Training with batch size {args.bsize} and gradient accumulation steps {args.gradient_accumulation_steps}"
            )
            logger.info(
                f"Total batch size: {args.bsize * accelerator.num_processes * args.gradient_accumulation_steps}"
            )
            best_checkpoints = []

        # Create infinite data loaders for step-based training
        train_iter = iter(train_loader)

        # Train model based on steps instead of epochs
        global_step = 0
        best_eval_loss = float("inf")
        early_stop_counter = 0

        # put the model into training mode
        model.train()

        # create progress bar and start training
        progress_bar = tqdm(
            range(args.steps),
            desc="Training",
            disable=not accelerator.is_local_main_process,
        )
        for step in progress_bar:
            # Get batch and handle end of iterator
            batch_dict = get_batch(
                train_iter,
                train_loader,
                accelerator,
                logger if accelerator.is_local_main_process else None,
            )

            # forward pass
            model_out = model(**batch_dict)

            # backward pass
            loss = model_out["loss"]
            accelerator.backward(loss / args.gradient_accumulation_steps)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # Gradient clipping
                if args.max_grad_norm is not None and args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                opt.step()
                if scheduler:
                    scheduler.step()
                opt.zero_grad()

            # Log loss
            gathered_loss = accelerator.gather(loss.detach()).mean()

            if (
                accelerator.is_local_main_process
                and global_step % args.logging_steps == 0
            ):
                with logging_redirect_tqdm():
                    logger.info(
                        f"Step: {global_step}, Loss: {gathered_loss.item():.7f}"
                    )

                progress_bar.set_postfix(
                    {
                        "lr": opt.param_groups[0]["lr"],
                    }
                )

            global_step += 1

            # Run evaluation at specified intervals - only on main process
            if global_step % args.eval_steps == 0:
                model.eval()

                if accelerator.is_local_main_process:
                    with logging_redirect_tqdm():
                        logger.info(f"Running evaluation at step {global_step}...")

                with torch.no_grad():
                    eval_losses = []
                    all_predictions = []
                    all_labels = []

                    for eval_step, batch_dict in enumerate(
                        tqdm(
                            val_loader,
                            desc="Evaluating",
                            total=len(val_dset)
                            // (args.bsize * accelerator.num_processes),
                            disable=not accelerator.is_local_main_process,
                        )
                    ):
                        if args.limit_eval_steps and eval_step >= args.limit_eval_steps:
                            break

                        # Use BF16 for evaluation to match training precision
                        with torch.amp.autocast(
                            "cuda", dtype=torch.bfloat16, enabled=True
                        ):
                            model_out = model(**batch_dict)

                        loss = model_out["loss"]
                        logits = model_out["logits"]

                        gathered_loss = accelerator.gather(loss).mean()
                        eval_losses.append(gathered_loss)

                        # Perform greedy CTC decoding
                        if accelerator.is_local_main_process:
                            predictions = greedy_ctc_decode(
                                logits, collator.vocab, blank_id=0
                            )
                            all_predictions.extend(predictions)
                            
                            # Get labels from text if available, otherwise decode from indices
                            if "labels_text" in batch_dict:
                                all_labels.extend(batch_dict["labels_text"])
                            else:
                                # Fallback: decode from label indices
                                inv_vocab = {v: k for k, v in collator.vocab.items()}
                                labels = batch_dict.get("labels", None)
                                if labels is not None:
                                    for label_ids in labels:
                                        # Filter out padding (0 in this case might be blank, not padding)
                                        label_text = "".join([inv_vocab.get(tid.item(), "") for tid in label_ids if tid.item() != 0])
                                        all_labels.append(label_text)

                    eval_loss = sum(eval_losses) / len(eval_losses)

                    # Save predictions to JSON
                    if accelerator.is_local_main_process:
                        predictions_dict = {
                            "step": global_step,
                            "eval_loss": eval_loss.item(),
                            "predictions": all_predictions,
                            "labels": all_labels,
                        }
                        
                        predictions_path = os.path.join(
                            args.out_dir, f"predictions_step_{global_step}.json"
                        )
                        with open(predictions_path, "w") as f:
                            json.dump(predictions_dict, f, indent=2)
                        
                        with logging_redirect_tqdm():
                            logger.info(f"Saved predictions to {predictions_path}")

                    # log the loss
                    if accelerator.is_local_main_process:
                        with logging_redirect_tqdm():
                            logger.info(
                                f"Step: {global_step}, Eval Loss: {eval_loss.item():.7f}"
                            )

                        unwrapped_model = accelerator.unwrap_model(model)

                        # Save checkpoint
                        ckpt_path = os.path.join(
                            args.out_dir, f"model_step_{global_step}.pt"
                        )
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
                            torch.save(
                                unwrapped_model.state_dict(),
                                os.path.join(args.out_dir, "model_best.pt"),
                            )
                            with logging_redirect_tqdm():
                                logger.info(
                                    f"New best model saved with eval loss: {eval_loss}"
                                )
                        else:
                            early_stop_counter += 1
                            with logging_redirect_tqdm():
                                logger.info(
                                    f"No improvement for {early_stop_counter} evaluations"
                                )

                        del unwrapped_model

                # Sync early stopping decision across processes
                should_stop = torch.tensor(
                    1.0 if early_stop_counter >= args.early_stopping_patience else 0.0,
                    device=accelerator.device,
                )
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


def parse_args():
    """parse command line arguments"""

    parser = argparse.ArgumentParser()

    io_group = parser.add_argument_group("Input and output related")
    io_group.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="path to the output directory to save experiment results, logs and checkpoints",
    )
    io_group.add_argument(
        "--data_path", type=str, required=True, help="path to the dataset"
    )

    model_group = parser.add_argument_group("Model definition related group")
    model_group.add_argument(
        "--from_pretrained", type=str, help="path to the pretrained CTC model"
    )
    model_group.add_argument(
        "--d_model", type=int, default=512, help="encoder model dimension"
    )
    model_group.add_argument(
        "--num_encoder_layers",
        type=int,
        default=6,
        help="number of transformer encoder layers",
    )
    model_group.add_argument(
        "--num_attention_heads", type=int, default=8, help="number of attention heads"
    )
    model_group.add_argument(
        "--intermediate_size",
        type=int,
        default=2048,
        help="feedforward intermediate size",
    )
    model_group.add_argument(
        "--dropout", type=float, default=0.1, help="dropout probability"
    )
    model_group.add_argument(
        "--max_source_positions",
        type=int,
        default=1500,
        help="maximum source positions",
    )

    # FDLP-specific arguments
    fdlp_group = parser.add_argument_group("FDLP feature extraction related")
    fdlp_group.add_argument(
        "--n_filters", type=int, default=80, help="Number of FDLP filters"
    )
    fdlp_group.add_argument(
        "--coeff_num", type=int, default=80, help="Number of modulation coefficients"
    )
    fdlp_group.add_argument(
        "--coeff_range", type=str, default="1,80", help="Range of coefficients to preserve"
    )
    fdlp_group.add_argument(
        "--order", type=int, default=80, help="Order of FDLP model"
    )
    fdlp_group.add_argument(
        "--fduration", type=float, default=1.5, help="Duration of FDLP window in seconds"
    )
    fdlp_group.add_argument(
        "--frate", type=int, default=100, help="FDLP frame rate"
    )
    fdlp_group.add_argument(
        "--overlap_fraction", type=float, default=0.5, help="Overlap fraction for FDLP"
    )

    opt_group = parser.add_argument_group("Training and Optimization related")
    opt_group.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    opt_group.add_argument("--wdecay", type=float, default=0.0, help="weight decay")
    opt_group.add_argument(
        "--max_grad_norm", type=float, default=None, help="maximum gradient norm for clipping"
    )
    opt_group.add_argument(
        "--steps", type=int, default=10000, help="number of training steps"
    )
    opt_group.add_argument(
        "--eval_steps", type=int, default=100, help="evaluation frequency in steps"
    )
    opt_group.add_argument(
        "--limit_eval_steps",
        type=int,
        default=None,
        help="limit the number of evaluation steps",
    )
    opt_group.add_argument(
        "--logging_steps", type=int, default=10, help="logging steps"
    )
    opt_group.add_argument(
        "--warmup", type=int, default=1, help="number of warmup steps"
    )
    opt_group.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="gradient_accumulation_steps",
    )
    opt_group.add_argument(
        "--n_best", type=int, default=1, help="number of best checkpoints to keep"
    )
    opt_group.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="number of evaluations without improvement before early stopping",
    )

    opt_group.add_argument("--nj", type=int, default=0, help="number of workers")
    opt_group.add_argument("--shuffle", action="store_true", help="shuffle the data")
    opt_group.add_argument("--bsize", type=int, default=128, help="batch size")
    opt_group.add_argument("--seed", type=int, default=None, help="seed for rng init")

    data_group = parser.add_argument_group("Data related arguments")
    data_group.add_argument(
        "--train_split", type=str, default="train", help="Train split name"
    )
    data_group.add_argument(
        "--validation_split",
        type=str,
        default="validation",
        help="Validation split name",
    )
    data_group.add_argument(
        "--label_column",
        type=str,
        default="transcription",
        help="Column name for labels",
    )

    control_group = parser.add_argument_group("Trainer control related arguments")
    control_group.add_argument(
        "--do_train", action="store_true", help="Runs training in the trainer script"
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    main(args)
