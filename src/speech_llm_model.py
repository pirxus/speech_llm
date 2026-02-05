import torch
import sys
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from typing import Optional
from transformers import AutoModelForCausalLM, WhisperForConditionalGeneration

from connector import Connector

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: Optional[int], decoder_start_token_id: Optional[int]):
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
        self.speech_encoder = WhisperForConditionalGeneration.from_pretrained(
            speech_enc_id,
            torch_dtype=torch_dtype,
            attn_implementation="sdpa",
        ).model.encoder
        self.freeze_encoder()

        self.connector = Connector(
            self.speech_encoder.config.d_model, # FIXME not general
            self.llm.config.hidden_size,
            **connector_config,
            **kwargs,
        )

        # unfreeze the specified whisper layers
        #self.encoder_unfreeze_layers(enc_n_layers_to_unfreeze)

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
        #self.speech_encoder.load_state_dict(state_dict['speech_encoder'], **kwargs)
        self.connector.load_state_dict(state_dict['connector'], **kwargs)
        #state_dict['out_proj.weight'] = self.out_proj.weight.data
        #super(JointRetrieverQAModel, self).load_state_dict(state_dict, **kwargs)

    def encode_speech(self, speech_feats, attention_mask=None):
        # common for both branches
        speech_enc_out = self.speech_encoder(speech_feats, attention_mask=attention_mask).last_hidden_state

        if attention_mask is not None:
            if attention_mask.all():
                attention_mask = torch.ones(speech_enc_out.shape[:-1], device=speech_enc_out.device)
            else:
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
                #eos_token_id = 128009 # FIXME
                decoder_input_ids = shift_tokens_right(
                    labels, eos_token_id, self.llm.config.bos_token_id
                )

                # because of the way we compute loss, we don't need the shifted decoder_input_ids
                # NOTE: this may not be ideal, think about what this means for the prompt
                # suffix -- perhaps we need to enforce a space at the end of it?
                decoder_input_ids = decoder_input_ids[:,1:]

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
            conn_outputs = torch.hstack(
                (prefix_embeds, conn_outputs))

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
            conn_outputs = torch.hstack(
                (conn_outputs, suffix_embeds))

            if audio_attention_mask is not None:
                audio_attention_mask = torch.hstack(
                    (audio_attention_mask, prompt_suffix_mask))

        device = conn_outputs.device

        decoder_inputs_embeds = self.llm.get_input_embeddings()(decoder_input_ids)
        decoder_inputs_attn_mask = torch.ones_like(
            decoder_input_ids, device=device)

        decoder_inputs_embeds = torch.hstack(
            (conn_outputs, decoder_inputs_embeds))

        attention_mask = torch.hstack(
            (audio_attention_mask, decoder_inputs_attn_mask))

        decoder_outputs = self.llm(
            inputs_embeds=decoder_inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )

        logits = decoder_outputs.logits
        loss = None

        if labels is not None:
            labels = labels.to(logits.device)
            logits = logits[:, -labels.size(1):, :]
            # Shift so that tokens < n predict n
            shift_logits = logits.contiguous()
            shift_labels = labels.contiguous().to(logits.device)

            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction="mean")

            loss = loss_fct(
                shift_logits.view(-1, self.llm.config.vocab_size), shift_labels.view(-1))

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

        assert speech_feats is not None and speech_feats != speech_enc_out, "Either speech features or the encoder output has to be supplied"

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
            conn_outputs = torch.hstack(
                (prefix_embeds, conn_outputs))

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
            conn_outputs = torch.hstack(
                (conn_outputs, suffix_embeds))

            if audio_attention_mask is not None:
                audio_attention_mask = torch.hstack(
                    (audio_attention_mask, prompt_suffix_mask))

        decoder_outputs = self.llm.generate(
            inputs_embeds=conn_outputs,
            attention_mask=audio_attention_mask,
            **generate_kwargs,
        )

        return decoder_outputs
