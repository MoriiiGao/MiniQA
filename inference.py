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
                # load_lora(model, f'./{args.out_dir}/lora/{args.lora_name}_{args.hidden_size}.pth')
                load_lora(model, args.lora_path)
        else:
            transformers_model_path = '/root/models/Qwen_1.7'
            tokenizer = AutoTokenizer.from_pretrained(transformers_model_path)
            model = AutoModelForCausalLM.from_pretrained(transformers_model_path, trust_remote_code=True)
            if args.lora_name != 'None':
                apply_lora(model)
                # load_lora(model, f'./{args.out_dir}/lora/{args.lora_name}_{args.hidden_size}.pth')
                load_lora(model, args.lora_path)
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
                    'lora_deci': [
                        # "在面临多方向渗透威胁时，我军该如何识别敌主攻方向？",
                        # "敌方大规模部署电子战平台干扰我无人集群协同，我方应如何保障指挥与数据链稳定？",
                        # "我军若需在复杂城市环境中组织无人平台集群穿透敌防线，应如何部署与控制？",
                        # "面对敌方通过外线迂回突袭我西部要地的战略企图，我军应如何拆解应对方案？",
                        # "未来我军有对'台湾岛登陆作战'的军事问题，请将这个军事问题拆解成更细小的多个作战问题。",
                        # "未来我军有对'台湾岛登陆作战'的军事问题，请参考下面描述的例子，将这个军事问题拆解成更细小的多个作战问题。例如：'夏威夷岛登陆作战'可拆分为:如何从东部岛屿链登陆作战、如何从南部岛屿链登录作战。参考例子的同时，请加入一些自己的思考",
                        "请将“对台湾岛登陆作战”这一军事问题，参考“夏威夷岛登陆作战”多方向拆解的方式，细化为多个具体的作战问题。请从登陆方向、作战阶段、兵力协同、重点目标等多个维度进行详细分析，并结合你自己的判断补充创新要点。请以条目形式输出。"
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
    # test_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    test_mode = 0
    messages = []
    print("\n======= DeciMind Chat Engine =======\n")
    if test_mode == 0:
        for prompt in prompts:
            DeciMindChatEngine.setup_seed(random.randint(0, 2048))
            print(f"👶: {prompt}")
            messages = messages[-args.history_cnt:] if args.history_cnt else []
            messages.append({"role": "system", "content": "请您作为军事领域、作战领域、想定设计领域的专家"})
            messages.append({"role": "user", "content": prompt})
            response = engine.invoke(messages)
            # print('🤖️:', response)
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
            messages.append({"role": "system", "content": "请您作为军事领域、作战领域、想定设计领域的专家"})
            response = engine.invoke(messages)
            print('🤖️:', response)
            messages.append({"role": "assistant", "content": response})
            print("\n")

def parse_args():
    parser = argparse.ArgumentParser(description="Chat with DeciMind")
    parser.add_argument('--lora_name', default='lora_deci', type=str)
    parser.add_argument('--lora_path', default='/root/models/DeciMind/lora/lora_deci_640.pth', type=str)
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

