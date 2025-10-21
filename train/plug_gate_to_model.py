# from gate_train import load_model
from hf_qwen3_gate import Qwen3MoeGating, LlamaGating, GemmaGating, OlmoGating, Qwen3Gating
from transformers import AutoTokenizer
import torch, os

def main():
    thinking = False
    # turbo_gate_path = '/cpfs01/user/xiujian.zrj/cs_gate_train/models/opensource-turbo-nothink-gate-qwen3-controlfix-20k_95p_flores_2025-08-18-11:06:45'
    # turbo_gate_path = '/cpfs01/user/xiujian.zrj/cs_gate_train/models/opensource-turbo-nothink-gate-qwen3-controlfix-20k_95p_flores_2025-08-22-15:21:43'
    # turbo_gate_path = '/cpfs01/user/xiujian.zrj/cs_gate_train/models/gate-qwen-30b-norm-20k_95p_2025-08-28-08:59:51'
    # turbo_path = '/cpfs01/user/jiawei.lyt/ckpt/verl_checkpoints/lyt-rl-gen/qwen3-tpp-nothink-0721-distilled-data0706-recitex1-bothtrans-mixlangx2-GenRM-32B-sentcs-GSPO-ref-turbopp-LENGTH_FLIP_THRESHOLD1.3-LENGTH_FLIP_PROB0.75-REF_ANSWER_POSITION-A-expert-12k_bs512_minibs128_n8/global_step_60/actor_hf'

    # turbo_gate_path = '/cpfs01/user/xiujian.zrj/cs_gate_train/models/30b_think-gate-qwen3-controlfix-20k_95p_flores_2025-08-19-05:21:18'
    # turbo_path = '/cpfs01/user/jiawei.lyt/ckpt/verl_checkpoints/lyt-rl-gen/qwen3-tpp-thinking-fh0723-mkd035-distilled-data0706-recitex1-bothtrans-mixlangx2-GenRM-32B-sentcs-GSPO-ref-turbopp-THINK-FLIP1-2.4-LENGTH_FLIP_THRESHOLD1.3-LENGTH_FLIP_PROB0.75-REF_ANSWER_POSITION-A-expert-12k_bs512_minibs128_n8/global_step_90/actor_hf'
    gate_path = '/cpfs01/user/xiujian.zrj/cs_gate_train/models/gate-qwen-30b-think-norm-20k_95p_2025-09-01-12:07:35'
    model_path = '/cpfs01/user/jiawei.lyt/ckpt/verl_checkpoints/lyt-rl-gen/qwen3-tpp-thinking-fh0723-mkd035-distilled-data0706-recitex1-bothtrans-mixlangx2-GenRM-32B-sentcs-GSPO-ref-turbopp-THINK-FLIP1-2.4-LENGTH_FLIP_THRESHOLD1.3-LENGTH_FLIP_PROB0.75-REF_ANSWER_POSITION-A-expert-12k_bs512_minibs128_n8/global_step_90/actor_hf'

    # plus_gate_path = '/cpfs01/user/xiujian.zrj/cs_gate_train/models/plus-nothink-gate-qwen3-controlfix-20k_95p_flores_2025-08-17-09:01:10'
    # plus_path = '/cpfs02/user/jiawei.lyt/ckpt/verl_checkpoints/lyt-rl-gen/qwen3-ppp-nothink-fh0713-non_reason105step-rm_v1_addreason-GenRM-32B-sentcs-GSPO-ref-turbopp-static-USE_DYNAMIC_REF_ANSWER0-LENGTH_FLIP_THRESHOLD2.0-LENGTH_FLIP_PROB0.75-REF_ANSWER_POSITION-A-expert-12k_bs256_minibs128_n8/global_step_80/actor_hf'
    # gate_path = plus_gate_path
    # model_path = plus_path
    
    code_switch_head_weights = torch.load(os.path.join(gate_path, 'code_switch_head.pth'))
    code_switch_pre_weights = torch.load(os.path.join(gate_path, 'code_switch_pre.pth'))
    print(code_switch_head_weights.shape, code_switch_head_weights)
    print(code_switch_pre_weights.shape, code_switch_pre_weights)
    # model = load_model(model_path)
    model = Qwen3MoeGating.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    # load to the layer
    with torch.no_grad():
        model.code_switch_head.weight.copy_(code_switch_head_weights)
        model.code_switch_pre.weight.copy_(code_switch_pre_weights)
        print(model)
        if model.code_switch_head.bias is not None:
            model.code_switch_head.bias.zero_()
        if model.code_switch_pre.bias is not None:
            model.code_switch_pre.bias.zero_()
    model.save_pretrained(gate_path + '_plugged')
    os.system(f'cp /cpfs01/user/xiujian.zrj/cs_gate_train/models/tokenizer/* {gate_path + "_plugged"}')

def llama():
    # gate_path = '/cpfs01/user/xiujian.zrj/cs_gate_train/models/gate-llama3-8b-20k_95p_2025-08-24-11:41:02'
    gate_path = '/share/shmatikov/collin/language_confusion_paper/gate_weights/gate-llama3-8b-20k_95p_2025-08-24-11:41:02'
    model_path = 'meta-llama/Llama-3.1-8B-Instruct'
    
    code_switch_head_weights = torch.load(os.path.join(gate_path, 'code_switch_head.pth'))
    code_switch_pre_weights = torch.load(os.path.join(gate_path, 'code_switch_pre.pth'))
    print(code_switch_head_weights.shape, code_switch_head_weights)
    print(code_switch_pre_weights.shape, code_switch_pre_weights)
    # model = load_model(model_path)
    model = LlamaGating.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    # load to the layer
    with torch.no_grad():
        model.code_switch_head.weight.copy_(code_switch_head_weights)
        model.code_switch_pre.weight.copy_(code_switch_pre_weights)
        print(model)
        if model.code_switch_head.bias is not None:
            model.code_switch_head.bias.zero_()
        if model.code_switch_pre.bias is not None:
            model.code_switch_pre.bias.zero_()
    model.save_pretrained(gate_path + '_plugged')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(gate_path + '_plugged')


def gemma():
    # gate_path = '/cpfs01/user/xiujian.zrj/cs_gate_train/models/gate-llama3-8b-20k_95p_2025-08-24-11:41:02'
    # gate_path = '/cpfs01/user/xiujian.zrj/cs_gate_train/models/gate-gemma3-4b-20k_95p_2025-08-25-07:50:43'
    # model_path = '/cpfs01/user/xiujian.zrj/cs_gate_train/models/gemma3-4b'
    gate_path = '/share/shmatikov/collin/language_confusion_paper/gate_weights/gate-gemma3-12b-nonorm-20k_95p_2025-08-28-08:33:06'
    model_path = '/share/shmatikov/collin/language_confusion_paper/gate_weights/gemma3-12b'
    
    code_switch_head_weights = torch.load(os.path.join(gate_path, 'code_switch_head.pth'))
    code_switch_pre_weights = torch.load(os.path.join(gate_path, 'code_switch_pre.pth'))
    print(code_switch_head_weights.shape, code_switch_head_weights)
    print(code_switch_pre_weights.shape, code_switch_pre_weights)
    # model = load_model(model_path)
    model = GemmaGating.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    # load to the layer
    with torch.no_grad():
        model.code_switch_head.weight.copy_(code_switch_head_weights)
        model.code_switch_pre.weight.copy_(code_switch_pre_weights)
        print(model)
        if model.code_switch_head.bias is not None:
            model.code_switch_head.bias.zero_()
        if model.code_switch_pre.bias is not None:
            model.code_switch_pre.bias.zero_()
    model.save_pretrained(gate_path + '_plugged')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(gate_path + '_plugged')

def olmo():
    gate_path = '/cpfs01/user/xiujian.zrj/cs_gate_train/models/gate-olmo-32b-norm-20k_95p_2025-08-29-07:43:20'
    model_path = '/cpfs01/user/xiujian.zrj/cs_gate_train/models/OLMo-2-0325-32B-Instruct'
    
    code_switch_head_weights = torch.load(os.path.join(gate_path, 'code_switch_head.pth'))
    code_switch_pre_weights = torch.load(os.path.join(gate_path, 'code_switch_pre.pth'))
    print(code_switch_head_weights.shape, code_switch_head_weights)
    print(code_switch_pre_weights.shape, code_switch_pre_weights)
    exit()
    # model = load_model(model_path)
    model = OlmoGating.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    # load to the layer
    with torch.no_grad():
        model.code_switch_head.weight.copy_(code_switch_head_weights)
        model.code_switch_pre.weight.copy_(code_switch_pre_weights)
        print(model)
        if model.code_switch_head.bias is not None:
            model.code_switch_head.bias.zero_()
        if model.code_switch_pre.bias is not None:
            model.code_switch_pre.bias.zero_()
    model.save_pretrained(gate_path + '_plugged')
    # os.system(f'cp /cpfs01/user/xiujian.zrj/cs_gate_train/models/llama3-8b/*.json {gate_path + "_plugged"}')
    os.system(f"""cp {os.path.join(model_path, 'tokenizer.json')} {gate_path + "_plugged"}""")
    os.system(f"""cp {os.path.join(model_path, 'tokenizer_config.json')} {gate_path + "_plugged"}""")
    os.system(f"""cp {os.path.join(model_path, 'special_tokens_map.json')} {gate_path + "_plugged"}""")

def gptoss():
    from hf_qwen3_gate import GptOssGating
    gate_path = '/cpfs01/user/xiujian.zrj/cs_gate_train/models/gate-gpt-oss-20b-nonorm-20k_95p_2025-08-29-12:17:02'
    model_path = '/cpfs01/user/xiujian.zrj/cs_gate_train/models/gpt-oss-20b'
    
    code_switch_head_weights = torch.load(os.path.join(gate_path, 'code_switch_head.pth'))
    code_switch_pre_weights = torch.load(os.path.join(gate_path, 'code_switch_pre.pth'))
    print(code_switch_head_weights.shape, code_switch_head_weights)
    print(code_switch_pre_weights.shape, code_switch_pre_weights)
    # model = load_model(model_path)
    model = GptOssGating.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    # load to the layer
    with torch.no_grad():
        model.code_switch_head.weight.copy_(code_switch_head_weights)
        model.code_switch_pre.weight.copy_(code_switch_pre_weights)
        print(model)
        if model.code_switch_head.bias is not None:
            model.code_switch_head.bias.zero_()
        if model.code_switch_pre.bias is not None:
            model.code_switch_pre.bias.zero_()
    model.save_pretrained(gate_path + '_plugged')
    # os.system(f'cp /cpfs01/user/xiujian.zrj/cs_gate_train/models/llama3-8b/*.json {gate_path + "_plugged"}')
    os.system(f"""cp {os.path.join(model_path, 'tokenizer.json')} {gate_path + "_plugged"}""")
    os.system(f"""cp {os.path.join(model_path, 'tokenizer_config.json')} {gate_path + "_plugged"}""")
    os.system(f"""cp {os.path.join(model_path, 'special_tokens_map.json')} {gate_path + "_plugged"}""")
    os.system(f"""cp {os.path.join(model_path, 'generation_config.json')} {gate_path + "_plugged"}""")

def qwen_dense():
    gate_path = '/share/shmatikov/collin/language_confusion_paper/gate_weights/gate-qwen3-8b-nonorm-20k_95p_2025-09-02-02:50:25'
    model_path = 'Qwen/Qwen3-8B'
    
    code_switch_head_weights = torch.load(os.path.join(gate_path, 'code_switch_head.pth'))
    code_switch_pre_weights = torch.load(os.path.join(gate_path, 'code_switch_pre.pth'))
    print(code_switch_head_weights.shape, code_switch_head_weights)
    print(code_switch_pre_weights.shape, code_switch_pre_weights)
    # model = load_model(model_path)
    model = Qwen3Gating.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    # load to the layer
    with torch.no_grad():
        model.code_switch_head.weight.copy_(code_switch_head_weights)
        model.code_switch_pre.weight.copy_(code_switch_pre_weights)
        print(model)
        if model.code_switch_head.bias is not None:
            model.code_switch_head.bias.zero_()
        if model.code_switch_pre.bias is not None:
            model.code_switch_pre.bias.zero_()
    model.save_pretrained(gate_path + '_plugged')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(gate_path + '_plugged')

if __name__ == '__main__':
    # llama()
    gemma()
    # main()
    # olmo()
    # gptoss()
    # qwen_dense()
    # main()