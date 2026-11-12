"""HF CausalLM 自动接入: 权重目录 → prefill/decode 两个 DumpTarget。

移植自原 hf_llm_entry 的全部实测经验:
  - prefill 与 decode 分治, decode 是生成热路径;
  - decode 的 KV cache 会被原地追加 → make_inputs 每次重新 prefill 建
    新 cache, 保证各捕获层状态独立;
  - decode 的 attention_mask 用 torch.ones 整体构造（NPU 上 cat 混合
    格式操作数会被 aclnnCat 拒绝）;
  - decode 执行设备自动回退: 模型设备试跑一步失败（栈级问题如 KV cat
    格式回归）则整体搬 CPU——图结构设备无关。
"""

import torch

from . import DumpTarget, _safe


def _pick_device(spec):
    if spec and spec != "auto":
        return spec
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            return "npu"
    except Exception:  # noqa: BLE001
        pass
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _to_device_str(device):
    return f"{device}:0" if device in ("npu", "cuda") else device


def _build_tokenized_inputs(tokenizer, prompt, device):
    text = prompt
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
        except TypeError:  # 模板不支持 enable_thinking
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True)
        except Exception:  # noqa: BLE001  无模板时回落裸文本
            pass
    inputs = tokenizer(text, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


def build_targets(weights, prompt="请介绍一下你自己。", *, device="auto",
                  dtype=torch.bfloat16, attn_implementation="eager",
                  entries=("prefill", "decode")):
    """加载 HF CausalLM 并构建 prefill/decode 目标。返回 list[DumpTarget]。"""
    dev_type = _pick_device(device)
    dev = _to_device_str(dev_type)
    if dev_type == "npu":
        import torch_npu  # noqa: F401
        torch.npu.set_device(0)
        try:
            torch_npu.npu.config.allow_internal_format = False
        except Exception:  # noqa: BLE001
            pass

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(weights)
    model = AutoModelForCausalLM.from_pretrained(
        weights, torch_dtype=dtype,
        attn_implementation=attn_implementation).to(dev).eval()
    inputs = _build_tokenized_inputs(tokenizer, prompt, dev)
    input_ids, attention_mask = (inputs["input_ids"],
                                 inputs["attention_mask"])
    seq_len = input_ids.shape[1]
    print(f"[dump_tool] hf entry: {type(model).__name__} on {dev_type}, "
          f"seq_len={seq_len}", flush=True)

    # decode 执行设备探测: 失败则整体搬 CPU（图结构设备无关）
    if "decode" in entries and dev_type != "cpu":
        with torch.no_grad():
            pre = model(input_ids=input_ids, attention_mask=attention_mask,
                        use_cache=True)
            next_tok = pre.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            step_mask = torch.ones((1, seq_len + 1),
                                   dtype=attention_mask.dtype, device=dev)
            probe = _safe(
                "hf/decode 设备探测",
                lambda: model(input_ids=next_tok, attention_mask=step_mask,
                              past_key_values=pre.past_key_values,
                              use_cache=True))
        if probe is None:
            print("[dump_tool] 模型设备 decode 不可执行（栈级问题）→ "
                  "decode 目标回退 CPU", flush=True)
            model.to("cpu")
            inputs = {k: v.to("cpu") for k, v in inputs.items()}
            input_ids, attention_mask = (inputs["input_ids"],
                                         inputs["attention_mask"])
            dev = "cpu"

    targets = []

    if "prefill" in entries:
        targets.append(DumpTarget(
            name="prefill_forward", fn=model,
            make_inputs=lambda: ((),
                                 dict(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      use_cache=True))))

    if "decode" in entries:
        def _decode_inputs():
            """每次调用重新 prefill → 新 cache（各捕获层状态独立）。"""
            with torch.no_grad():
                pre = model(input_ids=input_ids,
                            attention_mask=attention_mask, use_cache=True)
                next_tok = pre.logits[:, -1, :].argmax(
                    dim=-1, keepdim=True)
                step_mask = torch.ones(
                    (1, seq_len + 1), dtype=attention_mask.dtype,
                    device=next_tok.device)
            return (), dict(input_ids=next_tok, attention_mask=step_mask,
                             past_key_values=pre.past_key_values,
                             use_cache=True)

        targets.append(DumpTarget(
            name="decode_step", fn=model, make_inputs=_decode_inputs))

    return targets
