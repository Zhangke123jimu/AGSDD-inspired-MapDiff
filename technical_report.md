# Technical Report of AGSDD-inspired MapDiff

## Introduction

Protein inverse folding aims to predict an amino-acid sequence from a given protein backbone structure. Recent neural approaches, including ProteinMPNN [1], Grade-IF [2], and MapDiff [3], have shown strong performance by combining residue-level geometric features with learned inter-residue communication.

MapDiff is a diffusion-based inverse folding model. It first uses an EGNN-based denoising network to produce a base prediction and then applies a mask-prior IPA network to refine high-uncertainty positions. This design makes MapDiff a strong baseline, but its residue representations are mainly learned through geometry-driven message passing and entropy-guided refinement.

AGSDD [4] introduces a semantic alignment (SA) module for diffusion-based inverse folding. Its motivation is that amino-acid types are not merely discrete class indices: they also encode biochemical, functional, and evolutionary regularities. AGSDD models this idea by using a learnable amino-acid type dictionary and attention-based alignment between residue hidden states and the 20 amino-acid type embeddings. The paper reports that, as denoising layers become deeper, residues tend to assign increasing attention to their true amino-acid types.

However, it is not obvious whether this semantic alignment idea can be directly transferred to other architectures. In AGSDD, semantic denoising is coupled with the model's geometric denoising and contextual aggregation design. MapDiff has a different decomposition: EGNN produces a base prediction, IPA refines masked positions, and entropy-based fusion combines the two outputs. Therefore, adding semantic alignment to MapDiff requires architecture-specific validation rather than a direct assumption of transferability.

This report studies an AGSDD-inspired semantic alignment extension of MapDiff. The extension is built on top of a dihedral-fixed MapDiff baseline, where the backbone dihedral feature calculation is fixed before adding semantic modules. The goals are:

1. Evaluate whether semantic alignment improves MapDiff on CATH 4.2.
2. Test whether the AGSDD-style attention alignment phenomenon can be reproduced in MapDiff.
3. Analyze how semantic alignment changes the output probability distribution of EGNN, IPA, and their fused predictions.

The resulting improvements are modest rather than dramatic, but the diagnostic results suggest that semantic alignment learns meaningful amino-acid type alignment and changes the final output distribution in a way that improves true-amino-acid ranking while keeping the probability distribution less sharply concentrated.

## Method

### MapDiff Baseline

MapDiff uses a discrete diffusion process over amino-acid sequences. During training, the clean sequence is denoted as $X_0^{aa}$, and $X_t^{aa}$ is the corrupted sequence at diffusion step $t$. Given the protein backbone geometry and $X_t^{aa}$, the denoising model predicts the original sequence $X_0^{aa}$.

The denoising model contains two main branches:

- **EGNN branch.** The EGNN branch uses backbone geometry and the noisy sequence to produce base logits over the 20 amino-acid types.
- **IPA branch.** The IPA branch is pretrained as a mask-prior model. During diffusion training and inference, EGNN first predicts the full sequence. Positions with high EGNN uncertainty are marked as masked positions, and IPA refines these positions using backbone geometry and the EGNN prediction as context.

The final logits are obtained by entropy-based fusion of EGNN and IPA outputs, following the original MapDiff design.

### Backbone Dihedral Feature Fix

Before adding semantic alignment, this project fixes several issues in MapDiff's backbone dihedral feature calculation. The fix includes converting dihedral angles from degrees to radians before applying sine and cosine, correcting the backbone phi-angle definition, and avoiding peptide-bond torsion features across likely chain breaks.

This change is treated as the geometry-fixed baseline for the semantic alignment experiments. The detailed motivation and validation are provided in the Appendix.

### Semantic Alignment Module

The semantic alignment module follows the core idea of AGSDD but is adapted to MapDiff's EGNN and IPA branches. For each semantic layer, the input hidden representation is denoted as $h \in \mathbb{R}^{N \times C}$, where $N$ is the number of residues and $C$ is the hidden dimension. Batch dimensions are omitted for simplicity.

Each branch owns a learnable amino-acid dictionary:

$$
H_s \in \mathbb{R}^{20 \times C}.
$$

In the current implementation, EGNN and IPA use separate dictionaries. Within each branch, the dictionary is shared across semantic layers, while each layer has its own query, key, value, gate, and output projections.

The hidden representation is projected to the query space:

$$
Q = h W_q^T + b_q,
$$

where $W_q \in \mathbb{R}^{C \times C}$ and $Q \in \mathbb{R}^{N \times C}$. The amino-acid dictionary is projected to key and value spaces:

$$
K = H_s W_k^T + b_k, \qquad V = H_s W_v^T + b_v,
$$

where $W_k, W_v \in \mathbb{R}^{C \times C}$ and $K,V \in \mathbb{R}^{20 \times C}$.

The semantic attention logits and probabilities are:

$$
A = \frac{QK^T}{\sqrt{C}}, \qquad P = \mathrm{softmax}(A).
$$

Here $A \in \mathbb{R}^{N \times 20}$ and $P \in \mathbb{R}^{N \times 20}$.

The semantic value update is:

$$
u = PV.
$$

where $u \in \mathbb{R}^{N \times C}$.

The AGSDD implementation also uses semantic information through a residual update rather than directly replacing the hidden representation with the attention-weighted semantic value. This implementation follows the same residual-update principle:

$$
g = sigmoid([h;u] W_g^T + b_g), \qquad \Delta = \mathrm{MLP}([h;u]),
$$

where $[h;u] \in \mathbb{R}^{N \times 2C}$, $W_g \in \mathbb{R}^{C \times 2C}$,  and $g,\Delta \in \mathbb{R}^{N \times C}$.

$$
h_{out} = h + g \odot \Delta.
$$

The output $h_{out}$ has shape $\mathbb{R}^{N \times C}$. The last linear layer of the adapter MLP is initialized to zero. This makes the semantic residual initially close to an identity update, while still allowing the semantic path to learn during training.

### Deep Semantic Embedding in MapDiff

The deep-embedded semantic version inserts semantic alignment into both MapDiff branches:

- In EGNN, semantic alignment is applied to the residue feature channel after each EGNN message-passing and feed-forward update. The coordinate channel is not directly modified by the semantic module.
- In IPA, semantic alignment is applied after each IPA block and transition update.

The semantic attention logits are supervised with a cross-entropy loss against the true amino-acid type. For EGNN, the semantic loss is computed over all valid residues:

$$
\mathcal{L}_{sem}^{EGNN}
= \frac{1}{L_E}\sum_{\ell=1}^{L_E}
\mathrm{CE}(A_{\ell}^{EGNN}, y),
$$

where $L_E$ is the number of EGNN semantic layers, $A_{\ell}^{EGNN}$ is the semantic attention logits at layer $\ell$, and $y$ is the true amino-acid label.

For IPA, the semantic loss is computed on masked positions, consistent with the IPA prediction loss:

$$
\mathcal{L}_{sem}^{IPA}
= \frac{1}{L_I}\sum_{\ell=1}^{L_I}
\mathrm{CE}(A_{\ell}^{IPA}[M], y[M]),
$$

where $M$ denotes the EGNN uncertainty mask.

The total training loss is:

$$
\mathcal{L}_{total} = \mathcal{L}_{EGNN} + \mathcal{L}_{IPA} + \lambda(\mathcal{L}_{sem}^{EGNN} + \mathcal{L}_{sem}^{IPA}).
$$

The final prediction and inference pipeline remain the same as MapDiff: EGNN and IPA logits are fused by the entropy-based fusion rule. The semantic attention logits are used for auxiliary supervision and hidden-state updates, not as an additional third prediction branch.

## Experiments

### Dataset and Evaluation Metrics

Experiments are conducted on CATH 4.2 using the same split and evaluation protocol as MapDiff. The main metrics are:

- **Median recovery rate.** The median sequence recovery rate across test proteins.
- **Perplexity.** Sequence-level perplexity computed from the predicted probability assigned to the ground-truth amino-acid sequence.

For mechanism analysis, additional residue-level metrics are used:

- **att_true.** Semantic attention probability assigned to the true amino-acid type.
- **rank_att_true.** Rank of the true amino-acid type among the 20 semantic attention weights. Lower is better.
- **att_entropy.** Shannon entropy of the semantic attention distribution over 20 amino-acid types. Lower values indicate a more concentrated semantic alignment.
- **mean_p_true.** Mean output probability assigned to the true amino-acid type.
- **top-k.** Fraction of residues where the true amino-acid type appears in the top-k predicted classes.
- **prob_margin.** Difference between the probability of the true amino acid and the highest probability among incorrect amino acids.

### Experimental Setting

The main semantic model is trained on top of the Dihedral-fixed MapDiff baseline. The mask-prior IPA pretraining follows the original MapDiff setting, and other hyperparameters are kept the same as the original MapDiff configuration unless otherwise specified. The main diffusion model is trained for 100 epochs on CATH 4.2.

The semantic loss weight is set to:

$$
\lambda = 0.5.
$$

This follows the spirit of AGSDD, where semantic supervision is trained at a scale comparable to the prediction objective rather than treated as a very small regularizer. In this implementation, the semantic loss is the sum of EGNN semantic loss and IPA semantic loss. Setting $\lambda = 0.5$ makes the EGNN prediction loss, IPA prediction loss, and two semantic-alignment losses approximately comparable in scale.

The main semantic model is the **Semantic-alignment & Dihedral-fixed** version reported in the main result table, where semantic alignment is deeply embedded across EGNN and IPA layers.

The experiments were completed on rented cloud GPU resources. Because full 100-epoch MapDiff training is expensive, the exploration of semantic variants and lambda values is not exhaustive. The main training runs use 2 x A100 GPUs to keep the comparison close to the original MapDiff setting.

Mechanism validation is run on the CATH 4.2 test set with ensemble size 50. Output distribution metrics are computed after ensemble-logit averaging. Semantic attention metrics are computed at the trajectory level for all, masked, and non-masked positions because the uncertainty mask may differ across trajectories.

## Results

### Main Results

| Version (CATH 4.2) | Parameters | Median recovery rate (%) | Perplexity |
|:--|--:|--:|--:|
| Reported MapDiff result (marginal prior) | 14.7M | 60.93 | 3.43 |
| Original reproduction | 14.7M | 60.97 | 3.54 |
| Dihedral-fixed | 14.7M | 61.23 | 3.54 |
| Semantic-alignment & Dihedral-fixed | 16.3M | 61.41 | 3.41 |

The Dihedral-fixed version gives a small recovery improvement over the original reproduction. Adding semantic alignment further moderately improves median recovery from 61.23% to 61.41% and reduces perplexity from 3.54 to 3.41.

Validation performance is reported at both 50 and 100 epochs:

| Version | Epoch | Validation median recovery rate (%) | Validation perplexity |
|:--|--:|--:|--:|
| Original reproduction | 50 | 57.25 | 3.91 |
| Original reproduction | 100 | 59.69 | 3.70 |
| Dihedral-fixed | 50 | 57.35 | 3.90 |
| Dihedral-fixed | 100 | 59.75 | 3.73 |
| Semantic-alignment & Dihedral-fixed | 50 | 56.78 | 3.93 |
| Semantic-alignment & Dihedral-fixed | 100 | 60.11 | 3.57 |

At epoch 50, the semantic version is slightly worse than the Dihedral-fixed version on validation recovery and perplexity. At epoch 100, this comparison reverses, and the semantic version becomes better on both metrics. This suggests that the semantic module may require longer optimization before its benefit becomes visible.

### Ablation Study

The final-only semantic version and the deep semantic version were compared on both validation and test sets. The final-only version is useful as a lightweight baseline, while the deep version follows AGSDD more closely by applying semantic alignment across multiple layers.

| Version | Validation median recovery rate (%) | Validation perplexity | Test median recovery rate (%) | Test perplexity |
|:--|--:|--:|--:|--:|
| Final-only semantic-alignment & Dihedral-fixed | 60.00 | 3.70 | 61.03 | 3.53 |
| Semantic-alignment & Dihedral-fixed | 60.11 | 3.57 | 61.41 | 3.41 |

The deep semantic version performs better than the final-only version on both recovery and perplexity. This supports the design choice of applying semantic alignment inside multiple EGNN and IPA layers rather than using it only as a final hidden-state adapter.

### Mechanism Analysis for Semantic Alignment

The mechanism analysis has two goals. First, it checks whether the AGSDD-style semantic alignment phenomenon can be reproduced in MapDiff. Second, it checks whether this internal alignment leads to sharper or better-ranked output distributions.

#### Semantic Alignment Replication

AGSDD reports that semantic attention to the true amino-acid type increases as the denoising network becomes deeper. The same trend appears in the Semantic-alignment & Dihedral-fixed checkpoint.

| Split | Layer | EGNN att_true | EGNN rank_att_true | EGNN att_entropy | IPA att_true | IPA rank_att_true | IPA att_entropy |
|:--|:--|--:|--:|--:|--:|--:|--:|
| all | L1 | 0.4508 | 3.1087 | 0.3939 | 0.1179 | 4.8522 | 2.5708 |
| all | L2 | 0.4671 | 2.8006 | 0.3499 | 0.1633 | 4.7546 | 2.4189 |
| all | L3 | 0.4747 | 2.7035 | 0.3299 | 0.2589 | 3.9232 | 2.0332 |
| all | L4 | 0.4778 | 2.6737 | 0.3245 | 0.3395 | 3.4045 | 1.7050 |
| all | L5 | 0.4816 | 2.6576 | 0.3342 | 0.4061 | 3.2132 | 1.4032 |
| all | L6 | 0.4834 | 2.6241 | 0.3023 | 0.4364 | 3.0730 | 1.2412 |
| mask | L1 | 0.1897 | 4.3074 | 0.6649 | 0.1120 | 5.6047 | 2.4777 |
| mask | L2 | 0.2135 | 3.8266 | 0.6718 | 0.1111 | 5.6421 | 2.4894 |
| mask | L3 | 0.2272 | 3.6599 | 0.6655 | 0.1348 | 5.1955 | 2.3599 |
| mask | L4 | 0.2333 | 3.6028 | 0.6693 | 0.1731 | 4.6877 | 2.1752 |
| mask | L5 | 0.2415 | 3.5702 | 0.6967 | 0.2026 | 4.4406 | 2.0272 |
| mask | L6 | 0.2443 | 3.5179 | 0.6481 | 0.2427 | 4.0974 | 1.8102 |
| non-mask | L1 | 0.6498 | 2.1946 | 0.1873 | 0.1223 | 4.2785 | 2.6418 |
| non-mask | L2 | 0.6605 | 2.0183 | 0.1044 | 0.2031 | 4.0778 | 2.3651 |
| non-mask | L3 | 0.6635 | 1.9743 | 0.0739 | 0.3536 | 2.9531 | 1.7841 |
| non-mask | L4 | 0.6643 | 1.9652 | 0.0615 | 0.4664 | 2.4261 | 1.3464 |
| non-mask | L5 | 0.6647 | 1.9618 | 0.0578 | 0.5612 | 2.2773 | 0.9275 |
| non-mask | L6 | 0.6657 | 1.9425 | 0.0387 | 0.5842 | 2.2919 | 0.8074 |

Both EGNN and IPA show increasing attention to the true amino-acid type from shallow to deep layers. The true type also moves to a better attention rank, and attention entropy generally decreases. This supports the claim that the inserted semantic modules learn an amino-acid type alignment pattern similar to that reported in AGSDD.

#### Output Distribution Effect

The output distribution analysis compares the Dihedral-fixed checkpoint and the Semantic-alignment & Dihedral-fixed checkpoint. Metrics are computed after ensemble-logit averaging on the test set.

| Branch | Version | mean_p_true | top1 | top3 | top5 | mean_rank | prob_margin |
|:--|:--|--:|--:|--:|--:|--:|--:|
| EGNN | Dihedral-fixed | 0.5607 | 0.5963 | 0.8407 | 0.9167 | 2.2315 | 0.2580 |
| EGNN | Semantic-alignment & Dihedral-fixed | 0.5571 | 0.5987 | 0.8423 | 0.9176 | 2.2187 | 0.2613 |
| IPA | Dihedral-fixed | 0.4601 | 0.5595 | 0.7816 | 0.8640 | 2.7225 | 0.2180 |
| IPA | Semantic-alignment & Dihedral-fixed | 0.4502 | 0.5603 | 0.7814 | 0.8644 | 2.7168 | 0.2142 |
| Fused | Dihedral-fixed | 0.5492 | 0.6066 | 0.8511 | 0.9239 | 2.1479 | 0.2739 |
| Fused | Semantic-alignment & Dihedral-fixed | 0.5423 | 0.6082 | 0.8517 | 0.9239 | 2.1421 | 0.2733 |

The semantic model slightly improves top-1 accuracy and the mean rank of the true amino acid in EGNN, IPA, and fused predictions. However, it does not increase the mean probability assigned to the true amino acid. In fact, mean_p_true is slightly lower in the semantic model.

This distinction is important. Higher semantic attention to the true amino-acid type does not necessarily imply a sharper final output probability. The attention output is an intermediate representation update, not the final classifier probability. The gated residual adapter, subsequent network layers, and entropy-based EGNN-IPA fusion can redistribute probability mass. The observed effect is therefore better described as improved ranking stability rather than simple confidence amplification.

## Discussion

### Effect of Semantic Alignment

The experimental results show three related phenomena. First, Semantic-alignment & Dihedral-fixed moderately improves recovery and perplexity over the Dihedral-fixed baseline. Second, the semantic attention assigned to the true amino-acid type increases across layers, reproducing the main SA phenomenon reported in AGSDD. Third, the final output distribution is not simply sharpened: true-AA top-k and rank metrics improve, while mean_p_true and probability margin become slightly lower.

The key hypothesis behind this behavior is that different amino acids are not independent symbolic indices. Amino acids have biochemical and functional similarities, and after being transformed into high-dimensional hidden representations, related amino-acid types may still share partially similar representational structure. By exposing the model to a learnable amino-acid type dictionary, semantic alignment encourages the network to treat amino-acid labels as type-aware representations rather than purely discrete class IDs.

Under this interpretation, a slightly flatter output distribution is not necessarily negative. It may indicate that the model assigns probability more moderately among plausible amino-acid candidates instead of only increasing confidence in the top class. This may be useful beyond single top-1 recovery. In inverse folding, multiple different sequences can be compatible with the same backbone, and practical design workflows often sample multiple candidate sequences before downstream filtering or experimental validation. A model that keeps reasonable probability mass on plausible alternatives may provide better candidate sets for multi-sequence sampling. However, this sampling hypothesis is not directly verified in the current experiments and should be evaluated in future work.

### Why the Gain Is Modest

The modest gain is plausible for three reasons.

First, SA is not the dominant module even in AGSDD. In the AGSDD ablation study, semantic alignment contributes as one additional component rather than as a standalone source of large improvement. The result here is more conservative, but still consistent with SA providing a moderate architecture-dependent contribution.

Second, the current best checkpoint appears at the final training epoch, and the validation comparison between the Dihedral-fixed and Semantic-alignment & Dihedral-fixed versions changes from unfavorable at epoch 50 to favorable at epoch 100. This suggests that semantic alignment may need longer optimization to show its benefit. AGSDD reports 70,000 training steps. Using the reported batch size of 32 and a CATH train split of roughly 18k proteins, this corresponds to about 124-125 effective epochs. The current MapDiff experiment is trained for 100 epochs, so the semantic module may not yet be fully optimized under the current computational budget.

Third, MapDiff does not include AGSDD's contextual aggregation (CA) module. In AGSDD, semantic denoising is paired with additional contextual information from the geometric denoising pipeline. In MapDiff, especially inside EGNN, semantic alignment has less global context available when aligning residue hidden states to amino-acid type embeddings. This architectural difference may limit how much benefit SA can provide.

### Limitations

This study has several limitations.

- Only a limited set of semantic designs and lambda values were tested. Future experiments could include EGNN-only semantic alignment, IPA-only semantic alignment, additional lambda values such as 0.1, and different adapter initializations. These experiments were not fully explored because full MapDiff training is expensive on cloud GPU resources.
- The semantic model was mainly evaluated on CATH 4.2; broader dataset validation is still needed.
- The mechanism analysis can be extended. A useful next step is to test whether the semantic model assigns more probability mass to chemically or evolutionarily similar non-true amino acids. For example, one could measure BLOSUM-positive probability mass, or check whether cases with lower p_true are accompanied by higher probability on high-BLOSUM non-true amino acids.
- Multi-sequence sampling quality has not yet been directly evaluated. Since semantic alignment may preserve probability mass over plausible alternatives, future work should evaluate diversity, plausibility, and downstream structural compatibility of sampled sequence sets.
- The implementation differs from AGSDD because MapDiff does not include AGSDD's contextual aggregation module.

## Conclusion

This project introduces an AGSDD-inspired semantic alignment extension to a Dihedral-fixed MapDiff baseline. Mechanism analysis reproduces the AGSDD-style trend that residues increasingly attend to their true amino-acid type across layers. At the output level, semantic alignment mainly improves the ranking of the true amino acid while producing a slightly flatter probability distribution rather than sharply increasing true-AA probability.

Both AGSDD and this implementation show that semantic alignment can improve inverse folding models, but the magnitude of improvement is architecture-dependent. Adding SA to another model therefore requires careful architectural placement, training-budget consideration, and mechanism-level validation.

## Appendix

### Geometry-corrected Backbone Dihedral Feature Calculation

This project is built on top of a Dihedral-fixed MapDiff baseline. The same backbone dihedral feature correction was also prepared as a pull request to the original MapDiff repository.

#### Summary

While reproducing MapDiff, several issues were found in the backbone dihedral feature calculation in `get_node_features()` in `dataloader/cath_dataset.py`.

The geometry fix makes three changes:

1. Convert dihedral angles from degrees to radians before applying `np.sin` and `np.cos`.
2. Correct the phi-angle definition for residue `i` to:

$$
\phi_i = \mathrm{dihedral}(C_{i-1}, N_i, CA_i, C_i).
$$

3. Avoid computing peptide-bond-related dihedral features across likely chain breaks or non-connected adjacent residues by checking the distance between $C_i$ and $N_{i+1}$.

#### Motivation

In the original implementation, the angles returned by `dihedral()` appear to be measured in degrees, but they are directly passed to `np.sin` and `np.cos`. Since these functions expect radians, this can distort the geometric encoding.

The original phi-angle calculation uses:

```python
dihedral(c_coords[i], n_coords[i], c_alpha_coords[i], n_coords[i + 1])
```

The standard backbone phi angle for residue `i` is instead:

$$
\phi_i = \mathrm{dihedral}(C_{i-1}, N_i, CA_i, C_i).
$$

Finally, dihedral features involving adjacent residues are meaningful only when the corresponding residues are connected by a peptide bond. To make feature calculation more robust to chain breaks or missing residues, the geometry-fixed version checks peptide-bond connectivity using the $C_i$ to $N_{i+1}$ distance. Undefined torsions at termini or chain breaks are encoded as `(sin, cos) = (0, 0)`.

#### Validation

The original implementation and the Dihedral-fixed version were compared under the same setting:

- Dataset: CATH 4.2
- Prior: marginal prior
- Hardware: 2 x A100 GPUs
- Per-GPU batch size: 4
- Other settings: unchanged from the original configuration

| Version | Recovery | Perplexity |
|:--|--:|--:|
| Original reproduction | 60.97% | 3.54 |
| Dihedral-fixed | 61.23% | 3.54 |
| Reported MapDiff result | 60.93% | 3.43 |

The improvement is small, so this fix should not be interpreted as a major performance-oriented modification. The main motivation is to make the geometric node features biologically more consistent and avoid potentially misleading dihedral encodings.

## References

[1] Dauparas, J. et al. Robust deep learning-based protein sequence design using ProteinMPNN. Science 378, 49-56 (2022).

[2] Yi, K., Zhou, B., Shen, Y., Li, P., and Wang, Y. Graph denoising diffusion for inverse protein folding. NeurIPS (2023).

[3] Bai, P., Miljkovic, F., Liu, X. et al. Mask-prior-guided denoising diffusion improves inverse protein folding. Nature Machine Intelligence 7, 876-888 (2025).

[4] Wang, C., Zhou, Y., Wang, Z., Zhai, Z., Shen, J., and Zhang, K. Alternate Geometric and Semantic Denoising Diffusion for Protein Inverse Folding. In: Ribeiro, R. P. et al. Machine Learning and Knowledge Discovery in Databases. ECML PKDD 2025, Lecture Notes in Computer Science, vol. 16015. Springer, Cham (2026).
