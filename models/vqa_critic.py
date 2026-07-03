from models.modeling_mplug import BertConfig, BertModel, FusionModel
from models.visual_transformers import initialize_clip

import torch
from torch import nn
import torch.nn.functional as F


class MPLUGCritic(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.config_encoder = BertConfig.from_json_file(config["bert_config"])
        self.config_encoder.num_hidden_layers = self.config_encoder.text_encoder_layers
        self.config_fusion = BertConfig.from_json_file(config["bert_config"])
        self.visual_encoder, _ = initialize_clip(config)
        self.text_encoder = BertModel.from_pretrained(
            config["text_encoder"],
            config=self.config_encoder,
            add_pooling_layer=False,
        )
        self.fusion_encoder = FusionModel.from_pretrained(
            config["text_encoder"],
            config=self.config_fusion,
            add_pooling_layer=False,
        )
        self.large = False
        if self.config_encoder.hidden_size != config["vision_width"]:
            self.visn_fc = nn.Linear(config["vision_width"], self.config_encoder.hidden_size)
            self.visn_layer_norm = nn.LayerNorm(self.config_encoder.hidden_size, eps=1e-12)
            self.dropout = nn.Dropout(self.config_encoder.hidden_dropout_prob)
            self.large = True
        self.use_checkpoint = config.get("use_checkpoint", True)
        hidden = self.config_encoder.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(self.config_encoder.hidden_dropout_prob),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(self.config_encoder.hidden_dropout_prob),
            nn.Linear(hidden, num_labels),
        )

    def forward(self, image, text, labels=None):
        image = image.to(dtype=next(self.parameters()).dtype)
        image_embeds = self.visual_encoder.visual(
            image,
            skip_last_layer=True,
            use_checkpoint=self.use_checkpoint,
        )
        if self.large:
            image_embeds = self.dropout(self.visn_layer_norm(self.visn_fc(image_embeds)))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        text_output = self.text_encoder(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
        )
        fusion_output = self.fusion_encoder(
            encoder_embeds=text_output.last_hidden_state,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=False,
        )
        image_output, text_output = fusion_output
        pooled = torch.cat([image_output[:, 0], text_output[:, 0]], dim=-1)
        logits = self.classifier(pooled)
        if labels is None:
            return logits
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        return loss, logits
