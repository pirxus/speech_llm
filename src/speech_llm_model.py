import torch
import sys
import math
import json
import torch.nn as nn
import os
from torch.nn import CrossEntropyLoss, CTCLoss
from typing import Optional
from transformers import AutoModelForCausalLM, WhisperForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from connector import Connector, SinusoidalPositionalEmbedding


def shift_tokens_right(
    input_ids: torch.Tensor,
    pad_token_id: Optional[int],
    decoder_start_token_id: Optional[int],
):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id if decoder_start_token_id is not None else 0

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class SpeechLLMBase(nn.Module):
    def __init__(
        self,
        speech_enc=None,
        speech_enc_id=None,
        llm_id=None,
        llm=None,
        torch_dtype=torch.bfloat16,
        lora_r=None,
        lora_a=None,
        connector_config={},
        **kwargs,
    ):
        super(SpeechLLMBase, self).__init__()

        assert llm is not None or llm_id is not None, "Either llm or llm_id must be provided"
        assert speech_enc is not None or speech_enc_id is not None, "Either speech_enc or speech_enc_id must be provided"

        if llm is not None:
            self.llm = llm.cuda()
            self.llm_id = llm.config._name_or_path
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(llm_id, torch_dtype=torch_dtype).cuda()
        self.freeze_llm()

        # TODO make general and allow for other encoders
        if "whisper" in speech_enc_id:
            self.speech_encoder = WhisperForConditionalGeneration.from_pretrained(
                speech_enc_id,
                torch_dtype=torch_dtype,
                attn_implementation="sdpa",
            ).model.encoder
            self.freeze_encoder()

        else:
            self.speech_encoder = FBankEncoder(
                d_model=kwargs["d_model"],
                num_mel_bins=kwargs["num_mel_bins"],
            )

        self.connector = Connector(
            self.speech_encoder.config.get("d_model"),  # FIXME not general
            self.llm.config.hidden_size,
            **connector_config,
            **kwargs,
        )

        # unfreeze the specified whisper layers
        # self.encoder_unfreeze_layers(enc_n_layers_to_unfreeze)

    @classmethod
    def from_pretrained(cls, path, device="cpu", return_model_args=False):
        with open(os.path.join(path, "config.json"), "r") as f:
            model_args = json.load(f)

        # create the joint model
        model = cls(**model_args).to(device)

        # map_location = {"cuda:0": f"cuda:{process_index}"}
        map_location = device
        state_dict = torch.load(os.path.join(path, 'model_best.pt'), map_location=map_location)
        model.load_state_dict(state_dict)

        if return_model_args:
            return model, model_args

        return model

    def freeze_encoder(self):
        self.speech_encoder.requires_grad_(False)

    def freeze_llm(self):
        self.llm.requires_grad_(False)

    def encoder_unfreeze_layers(self, n):
        self.freeze_encoder()
        for i in range(1, n + 1):
            self.speech_encoder.layers[-i].requires_grad_(True)

    def state_dict(self, **kwargs):
        # FIXME -- adapt to the new scenario, think about how to store lora adapters
        return {
            "connector": self.connector.state_dict(**kwargs),
        }

    def load_state_dict(self, state_dict, **kwargs):
        # self.speech_encoder.load_state_dict(state_dict['speech_encoder'], **kwargs)
        self.connector.load_state_dict(state_dict["connector"], **kwargs)
        # state_dict['out_proj.weight'] = self.out_proj.weight.data
        # super(JointRetrieverQAModel, self).load_state_dict(state_dict, **kwargs)

    def encode_speech(self, speech_feats, attention_mask=None):
        # common for both branches
        speech_enc_out = self.speech_encoder(speech_feats, attention_mask=attention_mask).last_hidden_state

        if attention_mask is not None:
            if attention_mask.all():
                attention_mask = torch.ones(speech_enc_out.shape[:-1], device=speech_enc_out.device)
            else:
                try:
                    attention_mask = self.speech_encoder.downsample_attention_mask(attention_mask=attention_mask)
                    if attention_mask.shape[-1] >= speech_enc_out.shape[-2]:
                        attention_mask = attention_mask[:, : speech_enc_out.shape[-2]]
                    else:
                        diff = speech_enc_out.shape[-2] - attention_mask.shape[-1]
                        mask_appendix = attention_mask[...,-1].unsqueeze(1).repeat(1, diff)
                        attention_mask = torch.cat((attention_mask, mask_appendix), dim=1)
                except:
                    raise NotImplementedError("fix attention mask downsampling for the speech encoders")

        return speech_enc_out, attention_mask

    def forward(
        self,
        speech_feats=None,
        audio_attention_mask: Optional[torch.LongTensor] = None,
        prompt_prefix_ids: Optional[torch.LongTensor] = None,
        prompt_prefix_mask: Optional[torch.LongTensor] = None,
        prompt_suffix_ids: Optional[torch.LongTensor] = None,
        prompt_suffix_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        if labels is not None:
            # we don't want a bos token at the beginning of the labels
            if labels[0, 0] == self.llm.config.bos_token_id:
                labels = labels[:, 1:]

            if decoder_input_ids is None:
                eos_token_id = self.llm.config.eos_token_id
                # eos_token_id = 128009 # FIXME
                decoder_input_ids = shift_tokens_right(
                    labels, eos_token_id, self.llm.config.bos_token_id
                )

                # because of the way we compute loss, we don't need the shifted decoder_input_ids
                # NOTE: this may not be ideal, think about what this means for the prompt
                # suffix -- perhaps we need to enforce a space at the end of it?
                decoder_input_ids = decoder_input_ids[:, 1:]

        # forward through the speech encoder and the connector
        speech_enc_out, audio_attention_mask = self.encode_speech(speech_feats, audio_attention_mask)
        conn_outputs, audio_attention_mask = self.connector(speech_enc_out, audio_attention_mask)

        batch_size = conn_outputs.shape[0]

        if audio_attention_mask is None:
            audio_attention_mask = torch.ones(
                (conn_outputs.shape[0], conn_outputs.shape[1]),
                device=conn_outputs.device,
                dtype=torch.long,
            )

        # prepend the prompt prefix to the connector output
        if prompt_prefix_ids is None:
            if self.llm.config.bos_token_id is not None:
                prompt_prefix_ids = (
                    torch.LongTensor([[self.llm.config.bos_token_id]])
                    .repeat(batch_size, 1)
                    .to(conn_outputs.device)
                )
                prompt_prefix_mask = torch.ones_like(
                    prompt_prefix_ids, device=conn_outputs.device)
            else:
                prompt_prefix_ids = None

        else:
            # cut off the prefix eos token id
            if prompt_prefix_ids[0, -1] == self.llm.config.eos_token_id:
                print("WARNING: there was a trailing prompt prefix eos",
                      prompt_suffix_ids)
                prompt_prefix_ids = prompt_prefix_ids[..., :-1]
                prompt_prefix_mask = prompt_prefix_mask[..., :-1]

        # embed the prefix ids
        if prompt_prefix_ids is not None:
            prefix_embeds = self.llm.get_input_embeddings()(prompt_prefix_ids)
            conn_outputs = torch.hstack((prefix_embeds, conn_outputs))

            if audio_attention_mask is not None:
                audio_attention_mask = torch.hstack(
                    (prompt_prefix_mask, audio_attention_mask))

        # append the prompt suffix
        if prompt_suffix_ids is not None:
            # cut off the bos token
            if (self.llm.config.bos_token_id is not None and
                    prompt_suffix_ids[0, 0] == self.llm.config.bos_token_id):
                print("WARNING: there was a trailing prompt suffix bos",
                      prompt_suffix_ids)
                prompt_suffix_ids = prompt_suffix_ids[..., 1:]
                prompt_suffix_mask = prompt_suffix_mask[..., 1:]

            # cut off the eos token
            if prompt_suffix_ids[0, -1] == self.llm.config.eos_token_id:
                print("WARNING: there was a trailing prompt suffix eos",
                      prompt_suffix_ids)
                prompt_suffix_ids = prompt_suffix_ids[..., :-1]
                prompt_suffix_mask = prompt_suffix_mask[..., :-1]

            # embed the suffix ids
            suffix_embeds = self.llm.get_input_embeddings()(prompt_suffix_ids)
            conn_outputs = torch.hstack((conn_outputs, suffix_embeds))

            if audio_attention_mask is not None:
                audio_attention_mask = torch.hstack(
                    (audio_attention_mask, prompt_suffix_mask))

        device = conn_outputs.device

        decoder_inputs_embeds = self.llm.get_input_embeddings()(decoder_input_ids)
        decoder_inputs_attn_mask = torch.ones_like(decoder_input_ids, device=device)

        decoder_inputs_embeds = torch.hstack((conn_outputs, decoder_inputs_embeds))

        attention_mask = torch.hstack((audio_attention_mask, decoder_inputs_attn_mask))

        decoder_outputs = self.llm(
            inputs_embeds=decoder_inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )

        logits = decoder_outputs.logits
        loss = None

        if labels is not None:
            labels = labels.to(logits.device)
            logits = logits[:, -labels.size(1) :, :]
            # Shift so that tokens < n predict n
            shift_logits = logits.contiguous()
            shift_labels = labels.contiguous().to(logits.device)

            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction="mean")

            loss = loss_fct(
                shift_logits.view(-1, self.llm.config.vocab_size), shift_labels.view(-1)
            )

        return {
            "logits": logits,
            "loss": loss,
        }

    @torch.no_grad()
    def generate(
        self,
        speech_feats: Optional[torch.LongTensor] = None,
        speech_enc_out: Optional[torch.LongTensor] = None,
        audio_attention_mask: Optional[torch.LongTensor] = None,
        prompt_prefix_ids: Optional[torch.LongTensor] = None,
        prompt_prefix_mask: Optional[torch.LongTensor] = None,
        prompt_suffix_ids: Optional[torch.LongTensor] = None,
        prompt_suffix_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ):
        assert speech_feats is not None and speech_feats != speech_enc_out, (
            "Either speech features or the encoder output has to be supplied"
        )

        if speech_enc_out is None:
            speech_enc_out, audio_attention_mask = self.encode_speech(speech_feats, audio_attention_mask)

        conn_outputs, audio_attention_mask = self.connector(speech_enc_out, audio_attention_mask) # FIXME: again, attention masks

        batch_size = conn_outputs.shape[0]

        if audio_attention_mask is None:
            audio_attention_mask = torch.ones(
                (conn_outputs.shape[0], conn_outputs.shape[1]),
                device=conn_outputs.device,
                dtype=torch.long,
            )

        # prepend the prompt prefix to the connector output
        if prompt_prefix_ids is None:
            if self.llm.config.bos_token_id is not None:
                prompt_prefix_ids = (
                    torch.LongTensor([[self.llm.config.bos_token_id]])
                    .repeat(batch_size, 1)
                    .to(conn_outputs.device)
                )
                prompt_prefix_mask = torch.ones_like(
                    prompt_prefix_ids, device=conn_outputs.device)
            else:
                prompt_prefix_ids = None

        else:
            # cut off the prefix eos token id
            if prompt_prefix_ids[0, -1] == self.llm.config.eos_token_id:
                print("WARNING: there was a trailing prompt prefix eos",
                      prompt_suffix_ids)
                prompt_prefix_ids = prompt_prefix_ids[..., :-1]
                prompt_prefix_mask = prompt_prefix_mask[..., :-1]

        # embed the prefix ids
        if prompt_prefix_ids is not None:
            prefix_embeds = self.llm.get_input_embeddings()(prompt_prefix_ids)
            conn_outputs = torch.hstack((prefix_embeds, conn_outputs))

            if audio_attention_mask is not None:
                audio_attention_mask = torch.hstack(
                    (prompt_prefix_mask, audio_attention_mask))

        # append the prompt suffix
        if prompt_suffix_ids is not None:
            # cut off the bos token
            if (self.llm.config.bos_token_id is not None and
                    prompt_suffix_ids[0, 0] == self.llm.config.bos_token_id):
                print("WARNING: there was a trailing prompt suffix bos",
                      prompt_suffix_ids)
                prompt_suffix_ids = prompt_suffix_ids[..., 1:]
                prompt_suffix_mask = prompt_suffix_mask[..., 1:]

            # cut off the eos token
            if prompt_suffix_ids[0, -1] == self.llm.config.eos_token_id:
                print("WARNING: there was a trailing prompt suffix eos",
                      prompt_suffix_ids)
                prompt_suffix_ids = prompt_suffix_ids[..., :-1]
                prompt_suffix_mask = prompt_suffix_mask[..., :-1]

            # embed the suffix ids
            suffix_embeds = self.llm.get_input_embeddings()(prompt_suffix_ids)
            conn_outputs = torch.hstack((conn_outputs, suffix_embeds))

            if audio_attention_mask is not None:
                audio_attention_mask = torch.hstack(
                    (audio_attention_mask, prompt_suffix_mask))

        decoder_outputs = self.llm.generate(
            inputs_embeds=conn_outputs,
            attention_mask=audio_attention_mask,
            **generate_kwargs,
        )

        return decoder_outputs


class FBankEncoder(nn.Module):
    def __init__(
        self,
        d_model=512,
        num_mel_bins=80,
        max_source_positions=1500,
    ):
        super(FBankEncoder, self).__init__()

        embed_dim = d_model
        self.config = {"d_model": embed_dim}
        self.num_mel_bins = num_mel_bins
        self.max_source_positions = max_source_positions
        self.embed_scale = math.sqrt(embed_dim)

        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)

        self.embed_positions = SinusoidalPositionalEmbedding(
            embed_dim, self.max_source_positions
        )

        self.attention_pooling = nn.MaxPool1d(2, stride=2)

    def downsample_attention_mask(self, attention_mask=None):
        attention_mask = self.attention_pooling(attention_mask.float()).long()
        return attention_mask

    def forward(self, x, attention_mask=None):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        x = self.embed_positions(x)
        return BaseModelOutput(x)


class CTCSpeechEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        num_mel_bins=80,
        num_encoder_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
        dropout=0.1,
        max_source_positions=1500,
        torch_dtype=torch.bfloat16,
    ):
        super(CTCSpeechEncoder, self).__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_mel_bins = num_mel_bins
        self.num_encoder_layers = num_encoder_layers
        self.max_source_positions = max_source_positions

        self.config = {"d_model": d_model}

        # Feature extraction layers (same as FBankEncoder)
        self.conv1 = nn.Conv1d(num_mel_bins, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)

        # Positional embeddings
        self.embed_positions = SinusoidalPositionalEmbedding(
            d_model, max_source_positions
        )

        # Transformer encoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_attention_heads,
                dim_feedforward=intermediate_size,
                dropout=dropout,
                activation="gelu",
                dtype=torch_dtype,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_encoder_layers,
        )

        # CTC projection head
        self.ctc_head = nn.Linear(d_model, vocab_size)

        # Attention mask pooling (2x downsampling from conv2)
        self.attention_pooling = nn.MaxPool1d(2, stride=2)

        # CTC loss
        self.ctc_loss = CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    def downsample_attention_mask(self, attention_mask=None):
        attention_mask = self.attention_pooling(attention_mask.float()).long()
        return attention_mask

    def encode(self, speech_feats=None, audio_attention_mask=None):
        """
        Forward pass through encoder layers (without CTC head).
        Returns encoder outputs compatible with FBankEncoder.
        """
        # Conv layers
        x = self.conv1(speech_feats)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)

        # Add positional embeddings
        x = self.embed_positions(x)

        # Downsample attention mask if provided
        if audio_attention_mask is not None:
            audio_attention_mask = self.downsample_attention_mask(audio_attention_mask)

            if audio_attention_mask.shape[-1] >= x.shape[-2]:
                audio_attention_mask = audio_attention_mask[:, : x.shape[-2]]
            else:
                diff = x.shape[-2] - audio_attention_mask.shape[-1]
                mask_appendix = audio_attention_mask[...,-1].unsqueeze(1).repeat(1, diff)
                audio_attention_mask = torch.cat((audio_attention_mask, mask_appendix), dim=1)

            # Convert to key_padding_mask format (True = masked)
            key_padding_mask = audio_attention_mask.eq(0)
        else:
            key_padding_mask = None

        # Transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)

        return x, audio_attention_mask

    def forward(
        self,
        speech_feats=None,
        audio_attention_mask=None,
        labels=None,
        label_lengths=None,
    ):
        """
        Forward pass with CTC loss computation.

        Args:
            speech_feats: Input features [batch, num_mel_bins, time]
            audio_attention_mask: Attention mask [batch, time]
            labels: Target labels [batch, label_seq_len]
            label_lengths: Length of each label sequence [batch]

        Returns:
            Dictionary with 'logits', 'loss', and 'encoder_output'
        """
        # Get encoder outputs
        encoder_output, audio_attention_mask = self.encode(
            speech_feats, audio_attention_mask
        )

        # CTC projection
        logits = self.ctc_head(encoder_output)

        loss = None
        if labels is not None and label_lengths is not None:
            # Compute input lengths from attention mask
            if audio_attention_mask is not None:
                input_lengths = audio_attention_mask.sum(dim=1)
            else:
                input_lengths = torch.full(
                    (logits.shape[0],),
                    logits.shape[1],
                    dtype=torch.long,
                    device=logits.device,
                )

            # Log probabilities for CTC
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            # CTC expects [time, batch, vocab]
            log_probs = log_probs.permute(1, 0, 2)

            # Compute CTC loss
            loss = self.ctc_loss(log_probs, labels, input_lengths, label_lengths)

        return {
            "logits": logits,
            "loss": loss,
            "encoder_output": encoder_output,
        }

    @classmethod
    def from_pretrained(cls, path, device="cpu", return_model_args=False):
        """Load a pretrained CTC encoder"""
        with open(os.path.join(path, "config.json"), "r") as f:
            model_args = json.load(f)

        model = cls(**model_args).to(device)

        map_location = device
        state_dict = torch.load(
            os.path.join(path, "model_best.pt"), map_location=map_location
        )
        model.load_state_dict(state_dict)

        if return_model_args:
            return model, model_args

        return model

    def save_pretrained(self, path):
        """Save model config and weights"""
        os.makedirs(path, exist_ok=True)

        config = {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "num_mel_bins": self.num_mel_bins,
            "num_encoder_layers": self.num_encoder_layers,
            "max_source_positions": self.max_source_positions,
        }

        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

        torch.save(self.state_dict(), os.path.join(path, "model_best.pt"))
