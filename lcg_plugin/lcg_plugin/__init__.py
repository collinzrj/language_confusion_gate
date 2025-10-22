# my_vllm_plugin/__init__.py

def register_plugin():
    from vllm import ModelRegistry

    gating_impl_file = 'merge_apply_top_k_p'
    # gating_impl_file = 'vllm_qwen3_gate_opt'

    # Register your custom model
    if "Qwen3Gating" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("Qwen3Gating", f"lcg_plugin.{gating_impl_file}:Qwen3Gating")
    print("Plugin Qwen3Gating registered")

    if "Qwen3MoeGating" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("Qwen3MoeGating", f"lcg_plugin.{gating_impl_file}:Qwen3MoeGating")
    print("Plugin Qwen3MoeGating registered")

    if "LlamaGating" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("LlamaGating", f"lcg_plugin.{gating_impl_file}:LlamaGating")

    if "GemmaGating" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("GemmaGating", f"lcg_plugin.{gating_impl_file}:GemmaGating")

    if "OlmoGating" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("OlmoGating", f"lcg_plugin.{gating_impl_file}:OlmoGating")

    if "GptOssGating" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("GptOssGating", f"lcg_plugin.{gating_impl_file}:GptOssGating")


    