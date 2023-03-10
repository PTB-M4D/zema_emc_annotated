# Changelog

<!--next-version-placeholder-->

## v0.7.1 (2023-01-24)
### Fix
* **CITATION:** Replace wrong license string for schema-compliance, thus to ensure interpretability ([`a62fc16`](https://github.com/PTB-M4D/zema_emc_annotated/commit/a62fc16d92f157e4fc5c8bd9dc6634720b5c1442))

**[See all commits in this version](https://github.com/PTB-M4D/zema_emc_annotated/compare/v0.7.0...v0.7.1)**

## v0.7.0 (2023-01-21)
### Feature
* **dataset:** Reintroduce strict hash checking, which can optionally be skipped ([`43360eb`](https://github.com/PTB-M4D/zema_emc_annotated/commit/43360eb405aafc468c3d4bd15794e95873c58ccc))
* **data_types:** Introduce one aggregated data type to specify the sample size of extracted data ([`b90af83`](https://github.com/PTB-M4D/zema_emc_annotated/commit/b90af8311a5c7a297e49b5dd10597bd2e77438b7))

### Fix
* **CITATION.cff:** Fix syntax and include more metadata ([`ea274a7`](https://github.com/PTB-M4D/zema_emc_annotated/commit/ea274a789f68cc574e95593e88f5a63a7151b74e))

### Documentation
* **dataset:** Improve docstring of ZeMASamples ([`e06e793`](https://github.com/PTB-M4D/zema_emc_annotated/commit/e06e7938bb336701446b53466260df3fc81e5fba))

**[See all commits in this version](https://github.com/PTB-M4D/zema_emc_annotated/compare/v0.6.0...v0.7.0)**

## v0.6.0 (2023-01-21)
### Feature
* **DOI:** Introduce DOI into metadata and README badge ([`9e183e1`](https://github.com/PTB-M4D/zema_emc_annotated/commit/9e183e19d08f36b4116bcaf4797f0d669932a288))

**[See all commits in this version](https://github.com/PTB-M4D/zema_emc_annotated/compare/v0.5.0...v0.6.0)**

## v0.5.0 (2023-01-21)
### Feature
* **ReadTheDocs:** Introduce settings for ReadTheDocs ([`041e334`](https://github.com/PTB-M4D/zema_emc_annotated/commit/041e334e8d1fe5b71d313a459397f1d2e822dd0c))
* **CITATION.cff:** Introduce proper citation metadata ([`aea8e7a`](https://github.com/PTB-M4D/zema_emc_annotated/commit/aea8e7ad742fcdd4b8f74b5e30fc1417a4524592))

### Documentation
* **README:** Improve and extend README with Disclaimer and License sections as well as new links ([`78292f9`](https://github.com/PTB-M4D/zema_emc_annotated/commit/78292f94a4a86365d44e56525a7c26171807c665))
* **examples:** Improve and correct examples section ([`94ed4c0`](https://github.com/PTB-M4D/zema_emc_annotated/commit/94ed4c01241a6f4c3eaab8270ccfa7aada523d92))

**[See all commits in this version](https://github.com/PTB-M4D/zema_emc_annotated/compare/v0.4.0...v0.5.0)**

## v0.4.0 (2023-01-16)
### Feature
* **dataset:** Leave storage location specification to pooch to share data across local projects ([`dc7d7c9`](https://gitlab1.ptb.de/m4d/zema_emc_annotated/-/commit/dc7d7c9fbb61a3d0fe5e55e51d58b03a3d1ab6a5))

### Documentation
* **notebook:** Adapt read_dataset.ipynb to new implementation without manual cache location spec ([`18be701`](https://gitlab1.ptb.de/m4d/zema_emc_annotated/-/commit/18be7018f7d797293f58d1383760949175343b81))

## v0.3.0 (2023-01-15)
### Feature
* **dataset:** Introduce parameter to choose first sample to be extracted ([`125fe83`](https://gitlab1.ptb.de/m4d/zema_emc_annotated/-/commit/125fe8362dca4db3feeebf5d7b5c6030a93a3e2c))

## v0.2.1 (2022-12-30)
### Fix
* **dataset:** Remove strict hash checking to drastically increase performance ([`00a7a73`](https://gitlab1.ptb.de/m4d/zema_emc_annotated/-/commit/00a7a7367eb69221b6ff395151a392080cef32c8))

## v0.2.0 (2022-12-30)
### Feature
* **dataset:** Turn dataset provider into class and fix normalization ([`e92c9bb`](https://gitlab1.ptb.de/m4d/zema_emc_annotated/-/commit/e92c9bb77b074bebef7ed91fb222361bdc633d06))
* **dataset:** Introduce scaler parameter to retrieve several datapoints from each cycle at once ([`30a5cf9`](https://gitlab1.ptb.de/m4d/zema_emc_annotated/-/commit/30a5cf99c27ad652c60552e0d8acb2ac251e696c))

### Documentation
* **dataset:** Improve description of allowed inputs ([`925bce6`](https://gitlab1.ptb.de/m4d/zema_emc_annotated/-/commit/925bce66f542769ae5e74b419e33cd458881fa70))

## v0.1.0 (2022-12-26)
### Feature
* Initial version of API, docs and project structure ([`283b1ba`](https://gitlab1.ptb.de/m4d/zema_emc_annotated/-/commit/283b1ba7afda549cb4b3d5a7d593b2bf4f2eff62))
