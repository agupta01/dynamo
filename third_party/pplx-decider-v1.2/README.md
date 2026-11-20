## Public Perplexity protocol and metadata

Repository: `perplexity-ai/pplx-decider-v1.2`

Pinned revision: `225c913ddc8761a8264f04f64548e1f97e443eb3`

License: **Apache License 2.0**. Verified using both the checkpoint LICENSE and
`source/pyproject.toml` (`license = "Apache-2.0"`).

Public sources used:

- [LICENSE](https://huggingface.co/perplexity-ai/pplx-decider-v1.2/blob/225c913ddc8761a8264f04f64548e1f97e443eb3/LICENSE)
- [NOTICE](https://huggingface.co/perplexity-ai/pplx-decider-v1.2/blob/225c913ddc8761a8264f04f64548e1f97e443eb3/NOTICE)
- [source/pyproject.toml](https://huggingface.co/perplexity-ai/pplx-decider-v1.2/blob/225c913ddc8761a8264f04f64548e1f97e443eb3/source/pyproject.toml)
- [source/src/pplx_decider/protocol.py](https://huggingface.co/perplexity-ai/pplx-decider-v1.2/blob/225c913ddc8761a8264f04f64548e1f97e443eb3/source/src/pplx_decider/protocol.py):
  protocol-v3 prompt text, marker neutralization, option descriptions, ordered
  bisection without truncation, and choice/score confidence formulas.
- [source/src/pplx_decider/model.py](https://huggingface.co/perplexity-ai/pplx-decider-v1.2/blob/225c913ddc8761a8264f04f64548e1f97e443eb3/source/src/pplx_decider/model.py):
  checkpoint calibration temperature, option-sliced softmax, and readout semantics.
- [decision_config.json](https://huggingface.co/perplexity-ai/pplx-decider-v1.2/blob/225c913ddc8761a8264f04f64548e1f97e443eb3/decision_config.json)
- [joint_config.json](https://huggingface.co/perplexity-ai/pplx-decider-v1.2/blob/225c913ddc8761a8264f04f64548e1f97e443eb3/joint_config.json)
- [chat_template.jinja](https://huggingface.co/perplexity-ai/pplx-decider-v1.2/blob/225c913ddc8761a8264f04f64548e1f97e443eb3/chat_template.jinja)

Changes relative to that public reference: CPU-only text tokenization; no
PyTorch, images, model loading, or GPU execution; a strict, text-only Decisions
request subset; ordered typed API answers; transport-free `plan`/`finish` methods;
and a caller-provided local metadata directory. No upstream source module is
imported or executed at runtime. Only local tokenizer metadata is loaded.

The upstream NOTICE is reproduced here:

> pplx-decider-v1.2
>
> Derived from Qwen/Qwen3.8-27B, revision
> 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0, under Apache License 2.0.

Unmodified upstream LICENSE and NOTICE copies are provided in
`third_party/pplx-decider-v1.2/`. Preserve those files, or integrate the notice
into the destination repository's third-party-notice mechanism, when moving the
codec into a PR.

## Decisions wire contract

Schema reference, not copied implementation:
[OpenAI Decisions create API](https://developers.openai.com/api/reference/resources/decisions/methods/create).

Consulted for ordered questions and answers, typed string/boolean choices,
zero-based score distributions, names, usage shape, and public schema limits.
No OpenAI implementation or SDK code is included or assigned a license here.
This codec supports only its documented text subset; it does not claim full
OpenAI API or confidence-calibration equivalence.
