import torch
import torch.nn as nn
import torch.nn.functional as F
from ...subNets import BertTextEncoder
from ...subNets.transformers_encoder.transformer import TransformerEncoder
from .Hierarchical import HhyperLearningEncoder,Transformer,CrossTransformer
from einops import repeat


class DHFN(nn.Module):
    def __init__(self, args):
        super(DHFN, self).__init__()
        if args.use_bert:
            self.text_model = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers,
                                              pretrained=args.pretrained)
        self.use_bert = args.use_bert
        dst_feature_dims, nheads = args.dst_feature_dim_nheads
        if args.dataset_name == 'mosi':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 500, 375
        if args.dataset_name == 'mosei':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
                print(f"🎯 DHFN Model: Using aligned data (50/50/50)")
            else:
                self.len_l, self.len_v, self.len_a = 50, 500, 500
                print(f"🎯 DHFN Model: Using unaligned data (50/500/500)")
        # Feature dimensions for three modalities: (feature_dim, num_samples, sequence_length)
        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims
        self.d_l = self.d_a = self.d_v = dst_feature_dims
        self.num_heads = nheads
        self.layers = args.nlevels
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.output_dropout = args.output_dropout
        self.text_dropout = args.text_dropout
        self.attn_mask = args.attn_mask
        combined_dim_low = self.d_a
        combined_dim_high = self.d_a
        combined_dim = (self.d_l + self.d_a + self.d_v) + self.d_l * 3
        output_dim = 1

        # 1. 1D Convolution layers for temporal modeling or dimension reduction
        self.conv_proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False)
        self.conv_proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False)
        self.conv_proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False)

        # 2. Modality-specific encoders
        self.encoder_s_l = self.get_network(self_type='l', layers=self.layers)
        self.encoder_s_v = self.get_network(self_type='v', layers=self.layers)
        self.encoder_s_a = self.get_network(self_type='a', layers=self.layers)

        # Modality-shared encoder
        self.encoder_c = self.get_network(self_type='l', layers=self.layers)

        # 3. Decoders for reconstructing three modalities
        self.decoder_l = nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=1, padding=0, bias=False)
        self.decoder_v = nn.Conv1d(self.d_v * 2, self.d_v, kernel_size=1, padding=0, bias=False)
        self.decoder_a = nn.Conv1d(self.d_a * 2, self.d_a, kernel_size=1, padding=0, bias=False)

        # 4. Projection layers for calculating cosine similarity between specific features
        self.proj_cosine_l = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1),
                                       combined_dim_low)
        self.proj_cosine_v = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1),
                                       combined_dim_low)
        self.proj_cosine_a = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1),
                                       combined_dim_low)

        # Alignment layers for shared features (c_l, c_v, c_a)
        self.align_c_l = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), combined_dim_low)
        self.align_c_v = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), combined_dim_low)
        self.align_c_a = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), combined_dim_low)

        # Self-attention modules for shared features
        self.self_attentions_c_l = self.get_network(self_type='l')
        self.self_attentions_c_v = self.get_network(self_type='v')
        self.self_attentions_c_a = self.get_network(self_type='a')

        # Projection layers for shared feature fusion
        self.proj1_c = nn.Linear(self.d_l * 3, self.d_l * 3)
        self.proj2_c = nn.Linear(self.d_l * 3, self.d_l * 3)
        self.out_layer_c = nn.Linear(self.d_l * 3, output_dim)

        # 5. FC layers for shared features (low-level)
        self.proj1_l_low = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), combined_dim_low)
        self.proj2_l_low = nn.Linear(combined_dim_low, combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1))
        self.out_layer_l_low = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), output_dim)
        self.proj1_v_low = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), combined_dim_low)
        self.proj2_v_low = nn.Linear(combined_dim_low, combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1))
        self.out_layer_v_low = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), output_dim)
        self.proj1_a_low = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), combined_dim_low)
        self.proj2_a_low = nn.Linear(combined_dim_low, combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1))
        self.out_layer_a_low = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), output_dim)

        # 6. FC layers for specific features (high-level)
        self.proj1_l_high = nn.Linear(128,128)
        self.proj2_l_high = nn.Linear(128, combined_dim_high)
        self.out_layer_l_high = nn.Linear(combined_dim_high, output_dim)

        # 7. Projection layers for fusion
        self.projector_l = nn.Linear(50, 1)
        self.projector_v = nn.Linear(self.d_v, self.d_v)
        self.projector_a = nn.Linear(self.d_a, self.d_a)
        self.projector_c = nn.Linear(3 * self.d_l, 3 * self.d_l)

        # 8. Final projection layers
        self.proj1 = nn.Linear(151, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)


        # ==================== Ablation Study Switches ====================
        # Ablation 1: Feature Disentanglement Module Switch
        # True: Use encoder_s (specific) and encoder_c (common) to separate features
        # False: No feature disentanglement, use unified encoder, s and c share same features
        self.use_disentanglement = getattr(args, 'use_disentanglement', True)

        # Ablation 2: Hierarchical Learning Module Switch
        self.use_hierarchical_learning = getattr(args, 'use_hierarchical_learning', True)

        # Ablation 3: Shared Feature Enhancement Module (FC + Self-Attention) Switch
        self.use_shared_enhance = getattr(args, 'use_shared_enhance', True)

        # Generate current ablation experiment name
        self.ablation_name = self._get_ablation_name()
        print(f"🔬 DHFN Ablation Config: {self.ablation_name}")

        # When not using feature disentanglement, use unified encoders
        if not self.use_disentanglement:
            self.encoder_unified_l = self.get_network(self_type='l', layers=self.layers)
            self.encoder_unified_v = self.get_network(self_type='v', layers=self.layers)
            self.encoder_unified_a = self.get_network(self_type='a', layers=self.layers)

        if self.use_hierarchical_learning:
            # Hierarchical learning fusion module
            self.h_hyper = nn.Parameter(torch.ones(1, args.token_len, args.token_dim))

            # BERT model for extracting text contextual representations
            # Output shape: [B, T_text, D_bert]
            self.bertmodel = BertTextEncoder(use_finetune=True, transformers='bert',
                                         pretrained=args.pretrained)

            self.proj_l = nn.Sequential(
                nn.Linear(args.l_input_dim, args.l_proj_dst_dim),
                Transformer(num_frames=args.l_input_length, save_hidden=False, token_len=args.token_length,
                            dim=args.proj_input_dim, depth=args.proj_depth, heads=args.proj_heads,
                            mlp_dim=args.proj_mlp_dim)
            )
            self.proj_a = nn.Sequential(
                nn.Linear(args.a_input_dim, args.a_proj_dst_dim),
                Transformer(num_frames=args.a_input_length, save_hidden=False, token_len=args.token_length,
                            dim=args.proj_input_dim, depth=args.proj_depth, heads=args.proj_heads,
                            mlp_dim=args.proj_mlp_dim)
            )
            self.proj_v = nn.Sequential(
                nn.Linear(args.v_input_dim, args.v_proj_dst_dim),
                Transformer(num_frames=args.v_input_length, save_hidden=False, token_len=args.token_length,
                            dim=args.proj_input_dim, depth=args.proj_depth, heads=args.proj_heads,
                            mlp_dim=args.proj_mlp_dim)
            )
            self.l_encoder = Transformer(num_frames=args.token_length, save_hidden=True, token_len=None,
                                         dim=args.proj_input_dim, depth=args.AHL_depth - 1, heads=args.l_enc_heads,
                                         mlp_dim=args.l_enc_mlp_dim)
            self.h_hyper_layer = HhyperLearningEncoder(dim=args.token_dim, depth=args.AHL_depth, heads=args.ahl_heads,
                                                       dim_head=args.ahl_dim_head, dropout=args.ahl_droup)
            self.fusion_layer = CrossTransformer(source_num_frames=args.token_len, tgt_num_frames=args.token_len,
                                                 dim=args.proj_input_dim, depth=args.fusion_layer_depth,
                                                 heads=args.fusion_heads, mlp_dim=args.fusion_mlp_dim)
        else:
            # Simple fusion alternative (w/o Hierarchical Learning)
            # Average pooling on specific features followed by concatenation
            # Output dimension matches proj1_l_high input dimension (128)
            self.simple_fusion = nn.Sequential(
                nn.Linear(self.d_l + self.d_a + self.d_v, 128),
                nn.ReLU(),
                nn.Dropout(args.output_dropout),
                nn.Linear(128, 128)  # Output 128-dim to match proj1_l_high input
            )

    def _get_ablation_name(self):
        """Generate name identifier for current ablation experiment"""
        if self.use_disentanglement and self.use_hierarchical_learning and self.use_shared_enhance:
            return "Full Model"

        ablations = []
        if not self.use_disentanglement:
            ablations.append("w/o Disentanglement")
        if not self.use_hierarchical_learning:
            ablations.append("w/o Hierarchical Learning")
        if not self.use_shared_enhance:
            ablations.append("w/o Shared Enhancement")

        return " + ".join(ablations) if ablations else "Full Model"

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def forward(self, text, audio, video, return_tsne_features=False):
        # Feature extraction
        if self.use_bert:
            text = self.text_model(text)   # [B, T_l, 768]
        x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training)  # → [B, 768, T_l]
        x_a = audio.transpose(1, 2) # [B, d_a_orig, T_a]
        x_v = video.transpose(1, 2) # [B, d_v_orig, T_v]

        # Projection: [B, d, T] -> [B, d_target, T'], where T' = T - k + 1
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.conv_proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.conv_proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.conv_proj_v(x_v)

        proj_x_l = proj_x_l.permute(2, 0, 1) # → [T_l', B, d_l]
        proj_x_v = proj_x_v.permute(2, 0, 1) # → [T_v', B, d_v]
        proj_x_a = proj_x_a.permute(2, 0, 1) # → [T_a', B, d_a]

        # ==================== Feature Disentanglement ====================
        if self.use_disentanglement:
            # Full version: Use specific encoder and common encoder to separate features
            s_l = self.encoder_s_l(proj_x_l)  # [T_l', B, d_l]
            s_v = self.encoder_s_v(proj_x_v)
            s_a = self.encoder_s_a(proj_x_a)

            c_l = self.encoder_c(proj_x_l)  # [T_l', B, d_l]
            c_v = self.encoder_c(proj_x_v)
            c_a = self.encoder_c(proj_x_a)
        else:
            # Ablation version: No feature disentanglement, use unified encoder
            # s and c share the same encoded features
            unified_l = self.encoder_unified_l(proj_x_l)  # [T_l', B, d_l]
            unified_v = self.encoder_unified_v(proj_x_v)
            unified_a = self.encoder_unified_a(proj_x_a)

            # Specific and common features are identical
            s_l, c_l = unified_l, unified_l
            s_v, c_v = unified_v, unified_v
            s_a, c_a = unified_a, unified_a

        s_l = s_l.permute(1, 2, 0) # → [B, d_l, T_l']
        s_v = s_v.permute(1, 2, 0)
        s_a = s_a.permute(1, 2, 0)

        c_l = c_l.permute(1, 2, 0) # → [B, d_l, T_l']
        c_v = c_v.permute(1, 2, 0)
        c_a = c_a.permute(1, 2, 0)
        c_list = [c_l, c_v, c_a]

        # Cosine + Alignment: Linear mapping from [B, d, T'] to [B, d * T'] flattened for FC layers
        c_l_sim = self.align_c_l(c_l.contiguous().view(x_l.size(0), -1))  # → [B, d_l]
        c_v_sim = self.align_c_v(c_v.contiguous().view(x_l.size(0), -1))
        c_a_sim = self.align_c_a(c_a.contiguous().view(x_l.size(0), -1))

        # ==================== Reconstruction Module (only when using disentanglement) ====================
        if self.use_disentanglement:
            recon_l = self.decoder_l(torch.cat([s_l, c_list[0]], dim=1))  # [B, 2*d_l, T_l'] → [B, d_l, T_l']
            recon_v = self.decoder_v(torch.cat([s_v, c_list[1]], dim=1))
            recon_a = self.decoder_a(torch.cat([s_a, c_list[2]], dim=1))

            recon_l = recon_l.permute(2, 0, 1)  # → [T_l', B, d_l]
            recon_v = recon_v.permute(2, 0, 1)
            recon_a = recon_a.permute(2, 0, 1)

            s_l_r = self.encoder_s_l(recon_l).permute(1, 2, 0)  # [B, d, T']
            s_v_r = self.encoder_s_v(recon_v).permute(1, 2, 0)
            s_a_r = self.encoder_s_a(recon_a).permute(1, 2, 0)
        else:
            # Ablation version: No reconstruction, use placeholders (to maintain output format consistency)
            recon_l = proj_x_l  # Use original projected features as placeholder
            recon_v = proj_x_v
            recon_a = proj_x_a
            s_l_r = s_l  # Placeholder, relevant losses won't be calculated during training
            s_v_r = s_v
            s_a_r = s_a

        s_l = s_l.permute(2, 0, 1) # [T', B, d]
        s_v = s_v.permute(2, 0, 1)
        s_a = s_a.permute(2, 0, 1)

        c_l = c_l.permute(2, 0, 1) # [T', B, d]
        c_v = c_v.permute(2, 0, 1)
        c_a = c_a.permute(2, 0, 1)

        # Shared feature enhancement module
        if self.use_shared_enhance:
            # Original: Flatten + FC enhancement + Self-attention + FC fusion
            hs_l_low = c_l.transpose(0, 1).contiguous().view(x_l.size(0), -1)
            repr_l_low = self.proj1_l_low(hs_l_low)
            hs_proj_l_low = self.proj2_l_low(
                F.dropout(F.relu(repr_l_low, inplace=True), p=self.output_dropout, training=self.training))
            hs_proj_l_low += hs_l_low

            hs_v_low = c_v.transpose(0, 1).contiguous().view(x_v.size(0), -1)
            repr_v_low = self.proj1_v_low(hs_v_low)
            hs_proj_v_low = self.proj2_v_low(
                F.dropout(F.relu(repr_v_low, inplace=True), p=self.output_dropout, training=self.training))
            hs_proj_v_low += hs_v_low

            hs_a_low = c_a.transpose(0, 1).contiguous().view(x_a.size(0), -1)
            repr_a_low = self.proj1_a_low(hs_a_low)
            hs_proj_a_low = self.proj2_a_low(
                F.dropout(F.relu(repr_a_low, inplace=True), p=self.output_dropout, training=self.training))
            hs_proj_a_low += hs_a_low

            c_l_att = self.self_attentions_c_l(c_l)
            if type(c_l_att) == tuple:
                c_l_att = c_l_att[0]
            c_l_att = c_l_att[-1]  # Get attention-enhanced features c_l_att, shape [B, d]

            c_v_att = self.self_attentions_c_v(c_v)
            if type(c_v_att) == tuple:
                c_v_att = c_v_att[0]
            c_v_att = c_v_att[-1]

            c_a_att = self.self_attentions_c_a(c_a)
            if type(c_a_att) == tuple:
                c_a_att = c_a_att[0]
            c_a_att = c_a_att[-1]

            c_fusion = torch.cat([c_l_att, c_v_att, c_a_att], dim=1)

            c_proj = self.proj2_c(
                F.dropout(F.relu(self.proj1_c(c_fusion), inplace=True), p=self.output_dropout,
                          training=self.training))
            c_proj += c_fusion
            logits_c = self.out_layer_c(c_proj)
        else:
            # Ablation version: Remove shared feature enhancement (no FC + Self-attention)
            # Only apply temporal average pooling on c_l / c_v / c_a, then directly concatenate and predict
            # c_* shape is [T', B, d], first average over time dimension T'
            c_l_att = torch.mean(c_l, dim=0)  # [B, d_l] - Create placeholder variable for t-SNE
            c_v_att = torch.mean(c_v, dim=0)  # [B, d_v]
            c_a_att = torch.mean(c_a, dim=0)  # [B, d_a]

            c_fusion = torch.cat([c_l_att, c_v_att, c_a_att], dim=1)  # [B, 3 * d_l]
            logits_c = self.out_layer_c(c_fusion)


        # Specific feature fusion: Hierarchical learning or simple fusion
        if self.use_hierarchical_learning:
            # Hierarchical learning fusion (full version)
            s_l = s_l.permute(1, 0, 2) # [T', B, d] -> [B, T', d]
            s_v = s_v.permute(1, 0, 2)
            s_a = s_a.permute(1, 0, 2)
            b = x_v.size(0)
            h_hyper = repeat(self.h_hyper, '1 n d -> b n d', b=b)

            h_v = self.proj_v(s_v)[:, :self.h_hyper.shape[1]]
            h_a = self.proj_a(s_a)[:, :self.h_hyper.shape[1]]
            h_l = self.proj_l(s_l)[:, :self.h_hyper.shape[1]]
            h_t_list = self.l_encoder(h_l)
            h_hyper = self.h_hyper_layer(h_t_list, h_a, h_v, h_hyper)
            feat = self.fusion_layer(h_hyper, h_t_list[-1])[:, 0]

            hs_proj_l_high = self.proj2_l_high(
                F.dropout(F.relu(self.proj1_l_high(feat), inplace=True), p=self.output_dropout, training=self.training))

            logits_l_high = self.out_layer_l_high(hs_proj_l_high)
        else:
            # Simple fusion alternative (w/o Hierarchical Learning)
            # Average pooling on specific features followed by concatenation
            s_l = s_l.permute(1, 0, 2)  # [T', B, d] -> [B, T', d]
            s_v = s_v.permute(1, 0, 2)
            s_a = s_a.permute(1, 0, 2)

            # Average pooling: [B, T', d] -> [B, d]
            s_l_pooled = torch.mean(s_l, dim=1)  # [B, d_l]
            s_v_pooled = torch.mean(s_v, dim=1)  # [B, d_v]
            s_a_pooled = torch.mean(s_a, dim=1)  # [B, d_a]

            # Concatenate and fuse
            s_concat = torch.cat([s_l_pooled, s_v_pooled, s_a_pooled], dim=1)  # [B, d_l + d_v + d_a]
            feat = self.simple_fusion(s_concat)  # [B, combined_dim_high]

            hs_proj_l_high = self.proj2_l_high(
                F.dropout(F.relu(self.proj1_l_high(feat), inplace=True), p=self.output_dropout, training=self.training))

            logits_l_high = self.out_layer_l_high(hs_proj_l_high)

        # Final fusion
        last = torch.sigmoid(self.projector_l(hs_proj_l_high))
        c_fusion = torch.sigmoid(self.projector_c(c_fusion))
        last_hs = torch.cat([last, c_fusion], dim=1)

        # Final prediction
        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs), inplace=True), p=self.output_dropout, training=self.training))

        output = self.out_layer(last_hs_proj)

        res = {
            'origin_l': proj_x_l,
            'origin_v': proj_x_v,
            'origin_a': proj_x_a,
            's_l': s_l,
            's_v': s_v,
            's_a': s_a,
            'c_l': c_l,
            'c_v': c_v,
            'c_a': c_a,
            's_l_r': s_l_r,
            's_v_r': s_v_r,
            's_a_r': s_a_r,
            'recon_l': recon_l,
            'recon_v': recon_v,
            'recon_a': recon_a,
            'c_l_sim': c_l_sim,
            'c_v_sim': c_v_sim,
            'c_a_sim': c_a_sim,
            'logits_l_hetero': logits_l_high,
            'logits_c': logits_c,
            'output_logit': output,
            # Ablation experiment flags: Used by training code to determine whether to calculate related losses
            'use_disentanglement': self.use_disentanglement,
            'use_hierarchical_learning': self.use_hierarchical_learning,
            'use_shared_enhance': self.use_shared_enhance,
            'ablation_name': self.ablation_name
        }

        return res