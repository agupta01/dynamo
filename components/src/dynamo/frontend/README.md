<!-- # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# Dynamo Frontend

The API gateway for serving LLM inference requests with OpenAI-compatible HTTP and KServe gRPC endpoints.

See [docs/components/frontend/](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/overview.md) for documentation.

## Opt-in Perplexity Decider v1.2

`POST /v1/decisions` serves the text-only predicate, choice, and score subset
using the existing pooling engine. No adapter service or HTTP request to
`/v1/pooling` is involved. Without `DYN_DECIDER_MODEL_DIR` the frontend does
not register the route or load a Decisions tokenizer.

The worker must already serve a **converted text-classifier checkpoint**, not
the unmodified Hugging Face checkpoint. Use the matching noncausal text
backbone and 255-class readout head from
[`perplexity-ai/pplx-decider-v1.2` at `225c913`](https://huggingface.co/perplexity-ai/pplx-decider-v1.2/tree/225c913ddc8761a8264f04f64548e1f97e443eb3).
Start the vLLM worker with `--classify-worker --runner pooling --convert classify`.
Disable activation with `--pooler-config '{"pooling_type":"LAST","use_activation":false}'`.
Do not enable prefix caching on this noncausal model.

Configure the HTTP frontend:

- `DYN_DECIDER_MODEL_DIR`: local tokenizer, chat template, and Decisions
  protocol/calibration metadata matching the worker checkpoint.
- `DYN_DECIDER_SERVED_MODEL`: name registered by the pooling worker; defaults
  to `pplx-decider-v1.2`. The client-facing model is always `pplx-decider-v1.2`.
- `DECIDER_API_KEY_FILE`: file containing a random bearer key (32–4096
  printable, non-space ASCII bytes, optional final newline). The frontend
  reopens it per request. This protects **only** `/v1/decisions`; secure the
  transport and other frontend routes separately.

Example JSON body (send with `Content-Type: application/json` and bearer auth):

```json
{
  "model": "pplx-decider-v1.2",
  "input": "Every checkout request returns HTTP 503.",
  "questions": [{"type": "predicate", "instructions": "Is checkout unavailable?"}]
}
```

The frontend preserves question order, calibrates only valid classes, and
reports actual input tokens and zero generated tokens. It does not implement
generative decisions, multimedia inputs, streaming, or `/v1/systemone`.
