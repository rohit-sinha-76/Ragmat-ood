# Experimental Results Reference

This document compiles the exact experimental metrics and results recorded across all project phases.

---

## 1. Phase 3 Results: Tier-0 (Random Forest) Cosine vs. Cross-Attention

The following table lists the MAE values under different splits for `formation_energy` and `band_gap` using Magpie/CrystalNN descriptors:

### Target Property: formation_energy
- **IID Split**:
 - None (Baseline): 0.1063
 - Concat (Phase 1/2): 0.1078
 - Concat (Random): 0.1111
 - Cross-Attention: 0.1122
 - Cross-Attention (Random): 0.1179
- **Family-Out Split**:
 - None (Baseline): 0.2366
 - Concat (Phase 1/2): 0.2377
 - Concat (Random): 0.2390
 - Cross-Attention: 0.2500
 - Cross-Attention (Random): 0.2589
- **Element-Out Split**:
 - None (Baseline): 0.1805
 - Concat (Phase 1/2): 0.1827
 - Concat (Random): 0.1827
 - Cross-Attention: 0.1830
 - Cross-Attention (Random): 0.1900

### Target Property: band_gap
- **IID Split**:
 - None (Baseline): 0.2261
 - Concat (Phase 1/2): 0.2295
 - Concat (Random): 0.2384
 - Cross-Attention: 0.2380
 - Cross-Attention (Random): 0.2725
- **Family-Out Split**:
 - None (Baseline): 0.2529
 - Concat (Phase 1/2): 0.2544
 - Concat (Random): 0.2557
 - Cross-Attention: 0.2458
 - Cross-Attention (Random): 0.2987
- **Element-Out Split**:
 - None (Baseline): 0.3203
 - Concat (Phase 1/2): 0.3176
 - Concat (Random): 0.3243
 - Cross-Attention: 0.3141
 - Cross-Attention (Random): 0.3302

---

## 2. Phase 4 Results: Adaptive Gating Validation (H5 Evaluation)

PASS condition: gated MAE <= non_retrieval MAE for the same split and property.

| Job ID | Property | Split | Fusion | Non-Retrieval MAE | Gated MAE | H5 Status |
|---|---|---|---|---|---|---|
| P4v2_FE_1 | formation_energy | iid | concat | 0.10628 | 0.11172 | FAIL |
| P4v2_FE_2 | formation_energy | family_out | concat | 0.23656 | 0.23745 | FAIL |
| P4v2_FE_3 | formation_energy | element_out | concat | 0.18049 | 0.18619 | FAIL |
| P4v2_FE_4 | formation_energy | family_out | cross_attention | 0.23656 | 0.26268 | FAIL |
| P4v2_BG_1 | band_gap | iid | cross_attention | 0.22605 | 0.22780 | FAIL |
| P4v2_BG_2 | band_gap | family_out | cross_attention | 0.25290 | 0.23081 | PASS |
| P4v2_BG_3 | band_gap | element_out | cross_attention | 0.32029 | 0.29628 | PASS |
| P4v2_BG_4 | band_gap | family_out | concat | 0.25290 | 0.25292 | FAIL |

---

## 3. Phase 5 Results: Uncertainty Quantification

Evaluation under Conformal calibration (Target Coverage = 0.90) and Ensemble Variance:

### Conformal UQ Results

| Property | Split | Retrieval | Coverage | Mean Interval Width | MAE | RMSE | R2 |
|---|---|---|---|---|---|---|---|
| band_gap | element_out | OFF | 0.8415 | 1.3715 | 0.3203 | 0.6386 | 0.6364 |
| band_gap | element_out | true_neighbor | 0.8926 | 2.1647 | 0.3292 | 0.8127 | 0.4113 |
| band_gap | family_out | OFF | 0.8793 | 1.4259 | 0.2529 | 0.6068 | 0.6512 |
| band_gap | family_out | true_neighbor | 0.9232 | 2.1423 | 0.2368 | 0.7142 | 0.5169 |
| band_gap | iid | OFF | 0.8977 | 1.4018 | 0.2261 | 0.5136 | 0.8466 |
| band_gap | iid | true_neighbor | 0.8977 | 2.0176 | 0.2962 | 0.7248 | 0.6944 |
| formation_energy | element_out | OFF | 0.7662 | 0.5065 | 0.1805 | 0.2726 | 0.9189 |
| formation_energy | element_out | true_neighbor | 0.7770 | 0.5401 | 0.1878 | 0.2802 | 0.9143 |
| formation_energy | family_out | OFF | 0.6948 | 0.4747 | 0.2366 | 0.3901 | 0.8311 |
| formation_energy | family_out | true_neighbor | 0.7310 | 0.5214 | 0.2377 | 0.3992 | 0.8231 |
| formation_energy | iid | OFF | 0.8974 | 0.4904 | 0.1063 | 0.2148 | 0.9582 |
| formation_energy | iid | true_neighbor | 0.8979 | 0.5253 | 0.1140 | 0.2265 | 0.9536 |

---

## 4. Phase 6 Results: Tier 1 CGCNN Performance

### Stage 1: Base Encoder Performance (No Retrieval)

| Property | Split | MAE (all) | RMSE (all) | R2 (all) | MAE (low_ood) | MAE (high_ood) |
|---|---|---|---|---|---|---|
| band_gap | element_out | 0.4338 | 1.0595 | -0.0009 | 0.4289 | 0.7056 |
| band_gap | family_out | 0.1470 | 0.5317 | 0.7322 | 0.1236 | 0.5742 |
| band_gap | iid | 0.1728 | 0.4822 | 0.8647 | 0.1575 | 0.4714 |
| formation_energy | element_out | 0.5554 | 0.7487 | 0.3902 | 0.5502 | 0.6872 |
| formation_energy | family_out | 0.1444 | 0.2768 | 0.9150 | 0.1341 | 0.3886 |
| formation_energy | iid | 0.0651 | 0.1264 | 0.9856 | 0.0606 | 0.1521 |

### Stage 3: Retrieval Fusion Performance (CGCNN + RAG)

| Property | Split | Fusion | Mode | MAE (all) | RMSE | R2 | MRR | R@1 | R@10 | MAE (high_ood) |
|---|---|---|---|---|---|---|---|---|---|---|
| band_gap | element_out | cross_attention | random_control | 0.4368 | 1.0713 | -0.0234 | nan | nan | nan | 0.6978 |
| band_gap | element_out | cross_attention | true_neighbor | 0.4358 | 1.0658 | -0.0129 | 0.7268 | 0.7002 | 0.7791 | 0.7090 |
| band_gap | family_out | cross_attention | random_control | 0.1462 | 0.5320 | 0.7319 | nan | nan | nan | 0.5692 |
| band_gap | family_out | cross_attention | true_neighbor | 0.1460 | 0.5331 | 0.7308 | 0.8710 | 0.8392 | 0.9332 | 0.5746 |
| band_gap | iid | cross_attention | random_control | 0.1724 | 0.4826 | 0.8645 | nan | nan | nan | 0.4724 |
| band_gap | iid | cross_attention | true_neighbor | 0.1728 | 0.4810 | 0.8654 | 0.8522 | 0.8111 | 0.9348 | 0.4723 |
| formation_energy | element_out | concat | random_control | 0.5557 | 0.7483 | 0.3909 | nan | nan | nan | 0.6853 |
| formation_energy | element_out | concat | true_neighbor | 0.5661 | 0.7602 | 0.3714 | 0.2762 | 0.1825 | 0.5114 | 0.6898 |
| formation_energy | family_out | concat | random_control | 0.1417 | 0.2755 | 0.9158 | nan | nan | nan | 0.3917 |
| formation_energy | family_out | concat | true_neighbor | 0.1400 | 0.2618 | 0.9240 | 0.6580 | 0.5525 | 0.8738 | 0.3638 |
| formation_energy | iid | concat | random_control | 0.0622 | 0.1232 | 0.9863 | nan | nan | nan | 0.1493 |
| formation_energy | iid | concat | true_neighbor | 0.0605 | 0.1210 | 0.9868 | 0.8609 | 0.8009 | 0.9747 | 0.1460 |

---

## 5. Optimal Gating Performance

From the adaptive gating analysis (sweeping Mahalanobis detector thresholds):

- **formation_energy (element_out)**:
 - Base CGCNN MAE: 0.5554
 - RF Baseline MAE: 0.1805
 - Best Gated MAE: 0.1802 @ threshold 0.8 (100% routed to RF)
 - Recovery: 100.1% of performance gap closed
- **band_gap (element_out)**:
 - Base CGCNN MAE: 0.4338
 - RF Baseline MAE: 0.3203
 - Best Gated MAE: 0.3173 @ threshold 0.9 (88% routed to RF)
 - Recovery: 102.6% of performance gap closed

---

## 6. Zero-Shot Neighbor Imputation (ZSNI) Ablation

MAE metrics on element_out test sets across various seen neighbor numbers `k`:

- **formation_energy (base = 0.5554)**:
 - k = 1: 0.3319
 - k = 2: 0.2047
 - k = 3: 0.2689
 - k = 5: 0.3830
 - k = 7: 0.3622
 - k = 10: 0.3487
- **band_gap (base = 0.4338)**:
 - k = 1: 0.3497
 - k = 2: 0.2743
 - k = 3: 0.3021
 - k = 5: 0.3381
 - k = 7: 0.3454
 - k = 10: 0.3227

### ZSNI Pettifor + Electronegativity (formation_energy)
- k = 1: 0.3951
- k = 2: 0.4197
- k = 3: 0.3447
- k = 5: 0.3391
- k = 7: 0.3249
- k = 10: 0.3463

---

## 7. Results Files Checksums

The MD5 checksums of all output files in `results/` are:

- `bootstrap_cis_20260710_122535.json`: 12b19c70d43d0ac4896a349ae96c069b
- `bootstrap_cis_report.md`: ee95790d01f4c56b212ac387861d114a
- `conformal_zsni_summary.json`: 621dfc6e7f4160624221abd8f5010fbe
- `gating_analysis_20260710_123946.json`: 9a89869c817072e838d0155e82e9eb06
- `gating_analysis_20260710_125237.json`: 2ab25ab0f81a2fc7659734717c3fdc75
- `gating_analysis_20260710_130931.json`: 73a673237bc0315c64da1d2ba34ecc83
- `gating_final_report.md`: 4b9e369fa36c52a753cb3cba46077190
- `interpretability_data.json`: 85727052ff668323f91fd2c9962e4a58
- `interpretability_report.md`: aca3632f97aae1237c64b06ad2ed5769
- `phase3_final_report.md`: 5874d620dd8060cf8d60105326c9a6ba
- `phase4_final_report.md`: 1fcd724cc0c77b4a2d080cd4ae0947be
- `phase4_v2_final_report.md`: 1fcd724cc0c77b4a2d080cd4ae0947be
- `phase5_final_report.md`: bcaec2465fe616b376d191817043ecbd
- `phase5_retrieval_metrics.json`: 79e6e4c80a71a4f385bc0180c70c7002
- `phase6_base_band_gap_element_out_20260710_010437.json`: 714e98d37fee3efdc7de77e526907c7b
- `phase6_base_band_gap_family_out_20260709_220939.json`: eff218c8ec50c01fe1b08f7de99ab118
- `phase6_base_band_gap_iid_20260709_185923.json`: 7d8f1e545602c28b1150f0a034bed497
- `phase6_base_formation_energy_element_out_20260709_153920.json`: 1fdb0bd42c970ed0ec818578a16daabc
- `phase6_base_formation_energy_element_out_20260709_161418.json`: 55522a67db892519b7ef94dfbf1c660e
- `phase6_base_formation_energy_family_out_20260709_133920.json`: 4ab5538bdcc47ef96bd7fb92b27f7c84
- `phase6_base_formation_energy_family_out_20260709_161043.json`: 109c74d11a01ac37a4b379ff11ee6748
- `phase6_base_formation_energy_iid_20260709_090055.json`: 231c32e2b38459f118bb8462da389fc6
- `phase6_base_formation_energy_iid_20260709_095434.json`: 8ab30623499b4f77a3e14b449ffa9cd1
- `phase6_base_formation_energy_iid_20260709_100335.json`: 52a5893fbdd41c668a81bdbf0e9c2663
- `phase6_base_formation_energy_iid_20260709_105644.json`: 5e3eba65ed07f4af0a84806b13ee091a
- `phase6_base_formation_energy_iid_20260709_110824.json`: 2107165662d8cb0384cc2e9e0fcb7e5b
- `phase6_base_formation_energy_iid_20260709_113915.json`: c0e125749886464d445d428c9c90d533
- `phase6_base_formation_energy_iid_20260709_160701.json`: aa98d1f312778878e8fd4c24d138346e
- `phase6_final_report.md`: 639bc0024f43f05f6c73a74822ba9eaf
- `phase6_fusion_band_gap_element_out_cross_attention_random_control_20260710_041944.json`: 59813137f0dd5aab31315bbf44e29949
- `phase6_fusion_band_gap_element_out_cross_attention_true_neighbor_20260710_041557.json`: fb1f4b807edc22a168579abd46958721
- `phase6_fusion_band_gap_family_out_cross_attention_random_control_20260710_041145.json`: e66318bcc6fe3a0a4d8ce63ff97aaeba
- `phase6_fusion_band_gap_family_out_cross_attention_true_neighbor_20260710_040809.json`: c97630fcc3fed17de760b36ee1110ceb
- `phase6_fusion_band_gap_iid_cross_attention_random_control_20260710_040326.json`: b45d84313b49510c6665cf349a029f7c
- `phase6_fusion_band_gap_iid_cross_attention_true_neighbor_20260710_035958.json`: 9a64438823c09307c814966cf36630b7
- `phase6_fusion_formation_energy_element_out_concat_random_control_20260710_035533.json`: a2ce032f255989be8ecaf7b23ca76bb4
- `phase6_fusion_formation_energy_element_out_concat_true_neighbor_20260710_035111.json`: 631efa0238559786c33ad383dd96564b
- `phase6_fusion_formation_energy_family_out_concat_random_control_20260710_034543.json`: d0e73bdf54178cfef62782ded160b72e
- `phase6_fusion_formation_energy_family_out_concat_true_neighbor_20260710_034111.json`: ba03689cecaedce93bbc88d9450e24c7
- `phase6_fusion_formation_energy_iid_concat_random_control_20260710_024926.json`: ebd7aa0b42ffb9b9603811b7106c50b2
- `phase6_fusion_formation_energy_iid_concat_true_neighbor_20260710_024436.json`: bb5e81728714e969969bb69371a56ef6
- `predictions_P4v2_BG_1.csv`: 49ab5b77af5990c51cb229dacd2664e4
- `predictions_P4v2_BG_2.csv`: 5167cc51b06441b07bdeb1fdfc0496d6
- `predictions_P4v2_BG_3.csv`: 48df21d225996726f447f2fb2b2bba92
- `predictions_P4v2_BG_4.csv`: 58fe2ada6be79f050afb62e4a7d4b0e2
- `predictions_P4v2_FE_1.csv`: 738dc932212540a1a160d681ad54b648
- `predictions_P4v2_FE_2.csv`: 21f7e1bd0a403c129913d9bd0e69af57
- `predictions_P4v2_FE_3.csv`: 54da433f26cf20cbced19ebf42f9f19b
- `predictions_P4v2_FE_4.csv`: 78e74bce5db61cde278e71bc9fb1c309
- `predictions_phase4_band_gap_element_out_true_neighbor_cross_attention_gated.csv`: 0cba440d6408d9de262631d6348cd597
- `predictions_phase4_band_gap_family_out_true_neighbor_concat_gated.csv`: 3bb04ab88a0a958c18575f7cf2dc9d9d
- `predictions_phase4_band_gap_family_out_true_neighbor_cross_attention_gated.csv`: 7a7f7ae42419da19db7e75a282ccb6e4
- `predictions_phase4_band_gap_iid_true_neighbor_cross_attention_gated.csv`: c694e5764c9b67e5510e4659a93eb451
- `predictions_phase4_formation_energy_element_out_true_neighbor_concat_gated.csv`: 8740f5477d076d69f9e8b8eef7efc44e
- `predictions_phase4_formation_energy_family_out_true_neighbor_concat_gated.csv`: 4d09ffba084674214838ebbb09c6a6bc
- `predictions_phase4_formation_energy_family_out_true_neighbor_cross_attention_gated.csv`: ddffcb42f1c1f3a36e9d7f2fcde6fe0a
- `predictions_phase4_formation_energy_iid_true_neighbor_concat_gated.csv`: 365efebb5a5cde36f1a9fc4282a1528b
- `predictions_tier0_band_gap_element_out_none.csv`: dd5aaaceda0180b1f6d65be4e9604ccb
- `predictions_tier0_band_gap_element_out_random_control.csv`: a8f2f504b3701dd72135a60c233e29af
- `predictions_tier0_band_gap_element_out_random_control_cross_attention.csv`: 2e251e8a3621838aa5a1d9911c704512
- `predictions_tier0_band_gap_element_out_true_neighbor.csv`: a5d6bef7db79e41fa35d0191e662d4a6
- `predictions_tier0_band_gap_element_out_true_neighbor_cross_attention.csv`: d95ba11d123c8d6cab7519a25bd16f79
- `predictions_tier0_band_gap_family_out_none.csv`: 5c6104c507bb24f1bf1b85861065df3c
- `predictions_tier0_band_gap_family_out_random_control.csv`: 7f104cfa3c9cabd9a328e646491ca9a2
- `predictions_tier0_band_gap_family_out_random_control_cross_attention.csv`: e45a765231d796bf475bc6c581de8d40
- `predictions_tier0_band_gap_family_out_true_neighbor.csv`: 1f3f086dac1262a2d1f58e454b0553a7
- `predictions_tier0_band_gap_family_out_true_neighbor_cross_attention.csv`: 352655a27dec83c180e777ed62a1126c
- `predictions_tier0_band_gap_iid_none.csv`: 224151a018ac807984e4c36303775100
- `predictions_tier0_band_gap_iid_random_control.csv`: 1a28210e27e83e090b77080cc88dd52e
- `predictions_tier0_band_gap_iid_random_control_cross_attention.csv`: 9b05b96b5820639b5b9d3c47944a5816
- `predictions_tier0_band_gap_iid_true_neighbor.csv`: 981272e193c706626df1ae1f38a6b921
- `predictions_tier0_band_gap_iid_true_neighbor_cross_attention.csv`: ed644bf635bbaecf998b974c71c96cf5
- `predictions_tier0_formation_energy_element_out_none.csv`: 8ff4a219528bf7bf6a5e25559c866aea
- `predictions_tier0_formation_energy_element_out_random_control.csv`: a61bfc71edd2b73bfb60cec166072e00
- `predictions_tier0_formation_energy_element_out_random_control_cross_attention.csv`: d573ca3358918ab73f6e2ff92c2e50cd
- `predictions_tier0_formation_energy_element_out_true_neighbor.csv`: bd558ae39589ed6485246ae2a0a42273
- `predictions_tier0_formation_energy_element_out_true_neighbor_cross_attention.csv`: ba8d7009db81d2b2a6a275555a312607
- `predictions_tier0_formation_energy_family_out_none.csv`: 787b3e2fe96c9307b601cdacb4477194
- `predictions_tier0_formation_energy_family_out_random_control.csv`: a51214d3042a03ea07d93e132328d011
- `predictions_tier0_formation_energy_family_out_random_control_cross_attention.csv`: 865e336bf9ce6c8d0438bc6ccaf1abcd
- `predictions_tier0_formation_energy_family_out_true_neighbor.csv`: b49215a34b23bb3b41e963a82c82d2b9
- `predictions_tier0_formation_energy_family_out_true_neighbor_cross_attention.csv`: 03fb9d3fef9c27a84770373adc8780d8
- `predictions_tier0_formation_energy_iid_none.csv`: 5dcae2cdfec81dd68097978d2f94edb2
- `predictions_tier0_formation_energy_iid_random_control.csv`: 39c7b1ac8929f39e0e34e9a2de146fd8
- `predictions_tier0_formation_energy_iid_random_control_cross_attention.csv`: 84cd1ee2db39c5e72b0539c78ff00a18
- `predictions_tier0_formation_energy_iid_true_neighbor.csv`: 63ae70f3a5e3e0e937481968c344189d
- `predictions_tier0_formation_energy_iid_true_neighbor_cross_attention.csv`: c33ea9c48275ca4547bfa5dda2d2e1b4
- `predictions_tier1_band_gap_element_out_base.csv`: bc8fef7b1dc57a628a74d3a9d44718b4
- `predictions_tier1_formation_energy_element_out_base.csv`: ee6fc6b731f5e92ba485e7e8de1339b7
- `research_end_report.html`: fae30330087b8bfc918c51ebb9dd6a0b
- `results_P4v2_BG_1_20260707_023917.json`: 2af30517550d2674f96d229c39df3779
- `results_P4v2_BG_2_20260707_024119.json`: 304b14ee92ca34de7a22b319e68414e4
- `results_P4v2_BG_3_20260707_024333.json`: fb534533190ffe27543b85bf018c2332
- `results_P4v2_BG_4_20260707_024418.json`: 628f2de86eab3fe1a12a791382aa5a18
- `results_P4v2_FE_1_20260707_023338.json`: c46db23dc034b9c3343ea1db394e401f
- `results_P4v2_FE_2_20260707_023428.json`: a4b5caead5acdea59699feb3eb391990
- `results_P4v2_FE_3_20260707_023519.json`: de3b37d67eae3542a79d491720655f93
- `results_P4v2_FE_4_20260707_023721.json`: 3758e206c99a7cc09f73bc57a92cd384
- `results_phase4_band_gap_element_out_true_neighbor_cross_attention_gated_20260706_141615.json`: fef41a7060e9001ce40c288dd3414c70
- `results_phase4_band_gap_family_out_true_neighbor_concat_gated_20260706_140532.json`: f51c7198e279046dcda9ae19b0c1ca89
- `results_phase4_band_gap_family_out_true_neighbor_cross_attention_gated_20260706_141402.json`: aa549ab4dabea0a5140ee29f51334785
- `results_phase4_band_gap_iid_true_neighbor_cross_attention_gated_20260706_141201.json`: 2fb0c32a0bd03db1cfaf27f3f0c8e20b
- `results_phase4_formation_energy_element_out_true_neighbor_concat_gated_20260706_140125.json`: c6a276e5a34293b1fe5c95de2de38406
- `results_phase4_formation_energy_family_out_true_neighbor_concat_gated_20260706_140034.json`: fea3e7b15a2d6db4db864fbce96e42b0
- `results_phase4_formation_energy_family_out_true_neighbor_cross_attention_gated_20260706_141005.json`: c8260c1eac2d359d8b2587f77d823404
- `results_phase4_formation_energy_iid_true_neighbor_concat_gated_20260706_135942.json`: a53e5f308ffb41abec08c192b63d5412
- `results_phase5_conformal_band_gap_element_out_none_20260707_032345.json`: 3776f77ce95b652fcc9ddb62ce6f2895
- `results_phase5_conformal_band_gap_element_out_random_control_20260707_072345.json`: 43803d64db89308b3cd70cd1b6dc781b
- `results_phase5_conformal_band_gap_element_out_ret_20260707_032526.json`: 91a67d8c8394eadd2adbf4253b057dd0
- `results_phase5_conformal_band_gap_element_out_true_neighbor_20260707_072234.json`: 31e40b3168f4daee57be1ef871df8aac
- `results_phase5_conformal_band_gap_family_out_none_20260707_031712.json`: 9b8e9367adb9dd09b3f44abc75ffa969
- `results_phase5_conformal_band_gap_family_out_random_control_20260707_072145.json`: 3305c4e32e0d2908e1975fe20d762679
- `results_phase5_conformal_band_gap_family_out_ret_20260707_031821.json`: e8836594200f10fad400dcc60613c7a8
- `results_phase5_conformal_band_gap_family_out_true_neighbor_20260707_072041.json`: dc15ef9ea07264a981f9e00f2c5ef9cc
- `results_phase5_conformal_band_gap_iid_none_20260707_031104.json`: 0133ccfbd863c13001480fc3b07f76f1
- `results_phase5_conformal_band_gap_iid_random_control_20260707_071957.json`: 77c6504bb0e53123956dcab2d65cb857
- `results_phase5_conformal_band_gap_iid_ret_20260707_031212.json`: 81cb1f56148fd30a080ea9165afb3434
- `results_phase5_conformal_band_gap_iid_true_neighbor_20260707_071555.json`: 35c6785fde3cf30cd2a2ad172ecaf459
- `results_phase5_conformal_band_gap_iid_true_neighbor_20260707_071852.json`: 35c6785fde3cf30cd2a2ad172ecaf459
- `results_phase5_conformal_formation_energy_element_out_none_20260707_030930.json`: 75b2e92c1a67ce13b80035085f9823c7
- `results_phase5_conformal_formation_energy_element_out_ret_20260707_031017.json`: d1429109598ce326d49e1241d135ce70
- `results_phase5_conformal_formation_energy_family_out_none_20260707_030757.json`: 7525881659d288d01d389517f9705b1a
- `results_phase5_conformal_formation_energy_family_out_ret_20260707_030844.json`: 697f6f181866027a4c1bec488009716c
- `results_phase5_conformal_formation_energy_iid_none_20260707_030621.json`: 858c72c25e55d5561ed2d38ad6791773
- `results_phase5_conformal_formation_energy_iid_ret_20260707_030709.json`: 804da1359ef066efd9daa04d8ac25234
- `results_phase5_ensemble_variance_band_gap_element_out_none_20260707_032358.json`: 1cf3401dea696e3434161cb981b0ede4
- `results_phase5_ensemble_variance_band_gap_element_out_ret_20260707_033925.json`: 9dbc3946fbe868a047190d7a21b32dfa
- `results_phase5_ensemble_variance_band_gap_element_out_ret_20260707_034324.json`: acd4dcf09f94fa12312a5683a2aa1a5d
- `results_phase5_ensemble_variance_band_gap_family_out_none_20260707_031725.json`: 280a47fca3548ce372941682728d48eb
- `results_phase5_ensemble_variance_band_gap_family_out_ret_20260707_032331.json`: 28c1aca86325b15208df6c64a546df4e
- `results_phase5_ensemble_variance_band_gap_iid_none_20260707_031118.json`: 59a81040bf3cd3e07cb8e51eec89b147
- `results_phase5_ensemble_variance_band_gap_iid_ret_20260707_033413.json`: 140fa21cdb2df08726f8b2c3fc490760
- `results_phase5_ensemble_variance_formation_energy_element_out_none_20260707_030946.json`: b5c613a717ae9522272bc307510b1fcf
- `results_phase5_ensemble_variance_formation_energy_element_out_ret_20260707_031052.json`: a1ae32128de0909eba27d4540e639e8c
- `results_phase5_ensemble_variance_formation_energy_family_out_none_20260707_030812.json`: 77e2d6cd40f3375023358b3d76723722
- `results_phase5_ensemble_variance_formation_energy_family_out_ret_20260707_030916.json`: 3374abdde619587b33d30f12b12e9b49
- `results_phase5_ensemble_variance_formation_energy_iid_none_20260707_030637.json`: 26154ebcc651da2d0c366ea0df04032a
- `results_phase5_ensemble_variance_formation_energy_iid_ret_20260707_030742.json`: 461613da689874fe19c44a8eec267986
- `results_tier0_band_gap_element_out_none_20260705_144504.json`: b61e46624ddbdc1576ac574439efb78a
- `results_tier0_band_gap_element_out_random_control_20260705_144643.json`: 76aa49e8a93bb66b96b25e4f6f6cc1c7
- `results_tier0_band_gap_element_out_random_control_cross_attention_20260706_115116.json`: a95bfcfa9946c21d75284138479c277d
- `results_tier0_band_gap_element_out_true_neighbor_20260705_144553.json`: 20b8e06c2b32026136de922010b806cd
- `results_tier0_band_gap_element_out_true_neighbor_cross_attention_20260706_114200.json`: d4ecbe766a8986e6ffdfe2a6a959ad9e
- `results_tier0_band_gap_family_out_none_20260704_101237.json`: b996a5b35fe33fd9f82d17f122a5dca8
- `results_tier0_band_gap_family_out_random_control_20260705_144444.json`: 379b66bec0cfbf618e8b63656d66bbbf
- `results_tier0_band_gap_family_out_random_control_cross_attention_20260706_113032.json`: 8b95eae334640ec1726e0afa5642d833
- `results_tier0_band_gap_family_out_true_neighbor_20260705_044157.json`: 2b10ac2c37a154756cd7bdc315ef33bb
- `results_tier0_band_gap_family_out_true_neighbor_cross_attention_20260706_112030.json`: 27180aee39d794b949ac3ad4bbadf6d6
- `results_tier0_band_gap_iid_none_20260704_075444.json`: f0112508f02eb03ef647ebd2b20b6b85
- `results_tier0_band_gap_iid_random_control_20260704_092739.json`: d6930965c3a3018eff6cc34b2d6f29f3
- `results_tier0_band_gap_iid_random_control_cross_attention_20260706_110210.json`: 2781b9cbeb6d94362f3a0bff63b25729
- `results_tier0_band_gap_iid_true_neighbor_20260704_083939.json`: 28e2dfb94fd15d0a0b55e97eb1714468
- `results_tier0_band_gap_iid_true_neighbor_cross_attention_20260706_105452.json`: 13a18df078952f78ca996bb83b14136e
- `results_tier0_band_gap_iid_true_neighbor_cross_attention_20260707_055342.json`: dc2314f3b54b60d00aca95442f4e49e4
- `results_tier0_formation_energy_element_out_none_20260703_124356.json`: 10f2e95563ef517a7a2df0f5d95ba761
- `results_tier0_formation_energy_element_out_random_control_20260703_105111.json`: 309642ea776d2c3dac0aca76e1e57543
- `results_tier0_formation_energy_element_out_random_control_cross_attention_20260706_101849.json`: 0f37f5ab8d59a6dc3b7645d312960348
- `results_tier0_formation_energy_element_out_true_neighbor_20260703_153848.json`: 115e9f2d03f3be86271b42e8367f36ee
- `results_tier0_formation_energy_element_out_true_neighbor_cross_attention_20260706_100930.json`: 98447b3dcb0463d6e196eb4c69ebeee3
- `results_tier0_formation_energy_family_out_none_20260703_182942.json`: 9da9def81a1d49d94285360d0317a37f
- `results_tier0_formation_energy_family_out_random_control_20260703_120243.json`: e493c44e471df3186c9154f8f544a1bc
- `results_tier0_formation_energy_family_out_random_control_cross_attention_20260706_095356.json`: 674fec1cf3e483e4ba850094a02ae0b5
- `results_tier0_formation_energy_family_out_true_neighbor_20260704_041703.json`: e062a9e9a91f5387fc6c14bb333967fd
- `results_tier0_formation_energy_family_out_true_neighbor_cross_attention_20260706_094403.json`: c283e117c55538953f3c1ec24f12898f
- `results_tier0_formation_energy_iid_none_20260704_055524.json`: 9c1fd3c108310cebbb9a698bbb72c653
- `results_tier0_formation_energy_iid_random_control_20260704_063305.json`: 4ff87490493984cf371938f1977d78ae
- `results_tier0_formation_energy_iid_random_control_cross_attention_20260706_085631.json`: 067b3295ada11fbc2c8e75737ee1d2aa
- `results_tier0_formation_energy_iid_true_neighbor_20260704_061421.json`: 44a9ea65f302db4e35eacac166b6a9fb
- `results_tier0_formation_energy_iid_true_neighbor_cross_attention_20260706_081259.json`: a5c0ed931ca5c91fca93c5b7ce2a2032
- `zsni_ablation_results.json`: 874b52499b4a7df13314cf523b50c885
- `zsni_pettifor_summary.json`: 8c244fc7c8fda9b955190caf94ef7f9e

