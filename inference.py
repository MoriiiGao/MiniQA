import argparse
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_decimind import DeciMindConfig, DeciMindForCausalLM
from model.model_lora import apply_lora, load_lora
import asyncio

class DeciMindChatEngine:
    def __init__(self, args):
        self.args = args
        self.model, self.tokenizer = self.init_model()
        self.streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

    def init_model(self):
        args = self.args
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        if args.load == 0:
            moe_path = '_moe' if args.use_moe else ''
            modes = {0: 'pretrain', 1: 'full_sft', 2: 'dpo', 3: 'grpo'}
            ckp = f'{args.out_dir}/{modes[args.model_mode]}_{args.hidden_size}{moe_path}.pth'
            model = DeciMindForCausalLM(DeciMindConfig(
                hidden_size=args.hidden_size,
                num_hidden_layers=args.num_hidden_layers,
                use_moe=args.use_moe
            ))
            model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
            if args.lora_name != 'None':
                apply_lora(model)
                load_lora(model, f'./{args.out_dir}/lora/{args.lora_name}_{args.hidden_size}.pth')
        else:
            transformers_model_path = './DeciMind2'
            tokenizer = AutoTokenizer.from_pretrained(transformers_model_path)
            model = AutoModelForCausalLM.from_pretrained(transformers_model_path, trust_remote_code=True)
        print(f'DeciMind模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M(illion)')
        return model.eval().to(args.device), tokenizer

    def get_prompt_datas(self):
        args = self.args
        if args.model_mode == 0:
            prompt_datas = [
                '马克思主义基本原理',
                '人类大脑的主要功能',
                '万有引力原理是',
                '世界上最高的山峰是',
                '二氧化碳在空气中',
                '地球上最大的动物有',
                '杭州市的美食有'
            ]
        else:
            if args.lora_name == 'None':
                prompt_datas = [
                    '请介绍一下自己。',
                    '你更擅长哪一个学科？',
                    '鲁迅的《狂人日记》是如何批判封建礼教的？',
                    '我咳嗽已经持续了两周，需要去医院检查吗？',
                    '详细的介绍光速的物理概念。',
                    '推荐一些杭州的特色美食吧。',
                    '请为我讲解“大语言模型”这个概念。',
                    '如何理解ChatGPT？',
                    'Introduce the history of the United States, please.'
                ]
            else:
                lora_prompt_datas = {
                    'lora_identity': [
                        "你是ChatGPT吧。",
                        "你叫什么名字？",
                        "你和openai是什么关系？"
                    ]
                }
                prompt_datas = lora_prompt_datas[args.lora_name]
        return prompt_datas

    @staticmethod
    def setup_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def build_input(self, messages):
        args = self.args
        if args.model_mode != 0:
            new_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            new_prompt = self.tokenizer.bos_token + messages[-1]["content"]
        inputs = self.tokenizer(
            new_prompt,
            return_tensors="pt",
            truncation=True
        ).to(args.device)
        return inputs    

    def postprocess(self, response_ids, inputs):
        # 只取模型新生成部分
        return self.tokenizer.decode(
            response_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

    def invoke(self, messages):
        
        inputs = self.build_input(messages)
        generated_ids = self.model.generate(
            inputs["input_ids"],
            max_new_tokens=self.args.max_seq_len,
            num_return_sequences=1,
            do_sample=True,
            attention_mask=inputs["attention_mask"],
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            streamer=self.streamer,
            top_p=self.args.top_p,
            temperature=self.args.temperature
        )
        return self.postprocess(generated_ids, inputs)
    
    async def ainvoke(self, messages):

        # 支持异步流式输出
        inputs = self.build_input(messages)
        # 使用streamer作为回调，手动yield字符
        output_collector = []
        class AsyncStreamer(TextStreamer):
            def on_finalized_text(self, text, stream_end=False):
                output_collector.append(text)
        streamer = AsyncStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        _ = self.model.generate(
            inputs["input_ids"],
            max_new_tokens=self.args.max_seq_len,
            num_return_sequences=1,
            do_sample=True,
            attention_mask=inputs["attention_mask"],
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
            top_p=self.args.top_p,
            temperature=self.args.temperature
        )
        await asyncio.sleep(0)  # 让出事件循环以供streaming
        return "".join(output_collector)

def run_cli(args):
    engine = DeciMindChatEngine(args)
    prompts = engine.get_prompt_datas()
    test_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    messages = []
    print("\n======= DeciMind Chat Engine =======\n")
    if test_mode == 0:
        for prompt in prompts:
            DeciMindChatEngine.setup_seed(random.randint(0, 2048))
            print(f"👶: {prompt}")
            messages = messages[-args.history_cnt:] if args.history_cnt else []
            messages.append({"role": "user", "content": prompt})
            response = engine.invoke(messages)
            print('🤖️:', response)
            messages.append({"role": "assistant", "content": response})
            print("\n")
    else:
        while True:
            user_input = input("👶: ")
            if user_input.strip() in {"exit", "quit"}:
                break
            DeciMindChatEngine.setup_seed(random.randint(0, 2048))
            messages = messages[-args.history_cnt:] if args.history_cnt else []
            messages.append({"role": "user", "content": user_input})
            response = engine.invoke(messages)
            print('🤖️:', response)
            messages.append({"role": "assistant", "content": response})
            print("\n")

def parse_args():
    parser = argparse.ArgumentParser(description="Chat with DeciMind")
    parser.add_argument('--lora_name', default='None', type=str)
    parser.add_argument('--out_dir', default='/root/models/DeciMind', type=str)
    parser.add_argument('--temperature', default=0.85, type=float)
    parser.add_argument('--top_p', default=0.85, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--hidden_size', default=640, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=8192, type=int)
    parser.add_argument('--use_moe', default=True, type=bool)
    parser.add_argument('--history_cnt', default=0, type=int)
    parser.add_argument('--tokenizer_path', default="/root/MiniQA/model/PretrainTokenizer", type=str)
    parser.add_argument('--load', default=0, type=int, help="0: 原生torch权重，1: transformers加载")
    parser.add_argument('--model_mode', default=1, type=int,
                        help="0: 预训练模型，1: SFT-Chat模型，2: RLHF-Chat模型，3: RLAIF-Chat模型")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_cli(args)

