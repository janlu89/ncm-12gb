# Third-party code

This project builds on the Next Concept Model (NCM) reference implementation:

- Repository: https://github.com/HazyResearch/ncm
- License: MIT, Copyright (c) 2025 hermann kumbong
- Commit used for every experiment in this repository: `c70274273f489abc73832b2b98584a8483b6b203`

The upstream code is **not** included here. Our scripts import it from `third_party/ncm/` and never modify it.

## Getting the upstream code

From the root of this repository:

```
git clone https://github.com/HazyResearch/ncm third_party/ncm && git -C third_party/ncm checkout c70274273f489abc73832b2b98584a8483b6b203
```

## Authors' files

Every run reads `artifacts/tinystories/config.yaml` and the BPE vocabulary and merges next to it, even when
training from scratch; runs on the official tokenizer and the reference measurements also need the official
checkpoints. All of them come from the authors' Hugging Face repository `hazyresearch/ncm-checkpoints`
(folder `tinystories`); see the upstream README for the current links. They are not redistributed here.
