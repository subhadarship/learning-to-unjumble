import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from transformers import (
    RobertaConfig,
    BertPreTrainedModel,
    ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP,
    RobertaModel
)
from transformers.modeling_roberta import (
    ROBERTA_INPUTS_DOCSTRING
)

from transformers.file_utils import add_start_docstrings_to_callable

from transformers.activations import get_activation


class ElectraDiscriminatorPredictions(torch.nn.Module):
    """Prediction module for the discriminator, made up of two dense layers."""

    def __init__(self, config):
        super().__init__()

        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.dense_prediction = torch.nn.Linear(config.hidden_size, 1)
        self.config = config

    def forward(self, discriminator_hidden_states, attention_mask):
        hidden_states = self.dense(discriminator_hidden_states)
        hidden_states = get_activation(self.config.hidden_act)(hidden_states)
        logits = self.dense_prediction(hidden_states).squeeze()

        return logits


class RobertaForTokenDiscrimination(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        """
        self.lm_head = RobertaLMHead(config)  
        """

        self.discriminator_predictions = ElectraDiscriminatorPredictions(config)
        self.init_weights()

    """
    def get_output_embeddings(self):
        return self.lm_head.decoder
    """

    @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the token discrimination loss.
            Values should be in ``[-100, 0, 1]``
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, 1]``
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        loss (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            token discrimination loss
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    Examples::
        from transformers import RobertaTokenizer, RobertaForMaskedLM
        import torch
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForMaskedLM.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("cute, my dog is Hello", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([0, 1, 1, 1, 0], dtype=torch.long)  # demonstrative example
        outputs = model(input_ids, labels=labels)
        loss, prediction_scores = outputs[:2]
        """

        """
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if masked_lm_labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)
        """

        discriminator_hidden_states = self.roberta(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds
        )
        discriminator_sequence_output = discriminator_hidden_states[0]

        logits = self.discriminator_predictions(
            discriminator_sequence_output, attention_mask
        )

        outputs = (logits,)

        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1, discriminator_sequence_output.shape[1]) == 1
                active_logits = logits.view(-1, discriminator_sequence_output.shape[1])[active_loss]
                active_labels = labels[active_loss]
                loss = loss_fct(active_logits, active_labels.float())
            else:
                loss = loss_fct(logits.view(-1, discriminator_sequence_output.shape[1]), labels.float())

            outputs = (loss,) + outputs

        outputs += discriminator_hidden_states[1:]

        return outputs  # (loss), prediction_scores, (hidden_states), (attentions)
