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
(folder `tinystories`). They are not redistributed here.

Every result in this repository uses the release of 14-16 October 2026, revision
`d342ce8aaea58830eef3b8ad473664673e41f692` (download with `--revision d342ce8aaea58830eef3b8ad473664673e41f692`).
The authors replaced `tokenizer.pt`, `generator.pt` and `config.yaml` on 23 November 2026; that newer release was
not evaluated here. SHA-256 of the files used:

| File | Bytes | SHA-256 |
|---|---|---|
| `tinystories/tokenizer.pt` | 221,861,835 | `76e26003719db32dc3d542518951c286be07d6a5881f29ad5152d299764b0f81` |
| `tinystories/generator.pt` | 356,160,651 | `0acaf4f795390f9f7ccc8f84047867619a1d00e291681e299b855d7e563be6ff` |
| `tinystories/config.yaml` | 2,483 | `4009baccad5a159bfc12cd48f3da17c22c4a281b86f71f2fc70b7c1bbb62004b` |
