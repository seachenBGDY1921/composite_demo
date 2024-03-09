from __future__ import annotations

import os
import streamlit as st
import torch

from collections.abc import Iterable
from typing import Any, Protocol
from huggingface_hub.inference._text_generation import TextGenerationStreamResponse, Token
from transformers import AutoModel, AutoTokenizer, AutoConfig


from conversation import Conversation


from knowledge_service import KnowledgeService

TOOL_PROMPT = 'Answer the following questions as best as you can. You have access to the following tools:'

MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/chatglm3-6b')
PT_PATH = os.environ.get('PT_PATH', None)
PRE_SEQ_LEN = int(os.environ.get("PRE_SEQ_LEN", 128))
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)

@st.cache_resource
def get_client() -> Client:
    client = HFClient(MODEL_PATH, TOKENIZER_PATH, PT_PATH)
    return client

class Client(Protocol):
    def generate_stream(self,
                        system: str | None,
                        tools: list[dict] | None,
                        history: list[Conversation],
                        **parameters: Any
                        ) -> Iterable[TextGenerationStreamResponse]:
        ...

# ... 其他代码 ...

class HFClient(Client):
    def __init__(self, model_path: str, tokenizer_path: str, pt_checkpoint: str = None):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.knowledge_service = KnowledgeService()
        self.knowledge_service.init_knowledge_base()
        self.knowledge_base_path = self.knowledge_service.knowledge_base_path

        if pt_checkpoint is not None and os.path.exists(pt_checkpoint):
            config = AutoConfig.from_pretrained(
                model_path,
                trust_remote_code=True,
                pre_seq_len=PRE_SEQ_LEN
            )
            self.model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                config=config,
                device_map="auto").eval()
            prefix_state_dict = torch.load(os.path.join(pt_checkpoint, "pytorch_model.bin"))
            new_prefix_state_dict = {}
            for k, v in prefix_state_dict.items():
                if k.startswith("transformer.prefix_encoder."):
                    new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
            self.model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
        else:
            self.model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).quantize(4).cuda().eval()

    def generate_stream(self, system: str | None, tools: list[dict] | None, history: list[Conversation], **parameters) -> Iterable[TextGenerationStreamResponse]:
        chat_history = [{'role': 'system', 'content': system if not tools else TOOL_PROMPT}]

        if tools:
            chat_history[0]['tools'] = tools

        for conversation in history[:-1]:
            chat_history.append({
                'role': str(conversation.role).removeprefix('<|').removesuffix('|>'),
                'content': conversation.content,
            })

        query = history[-1].content
        role = str(history[-1].role).removeprefix('<|').removesuffix('|>')

        # 查询知识库
        knowledge_base_response = self.knowledge_service.search_knowledge_base(query)

        # 如果知识库中有答案，将其作为上下文加入模型输出
        if knowledge_base_response:
            chat_history.append({'role': 'knowledge_base', 'content': knowledge_base_response})
        else:
            # 如果知识库中没有答案，准备一个默认回答
            default_response = "抱歉哦，您问的问题超出了我的知识库范围，请重新提问。"

        # 生成模型输出
        for new_text, _ in stream_chat(
            self.model,
            self.tokenizer,
            query,
            chat_history,
            role,
            **parameters,
        ):
            word = new_text.removeprefix(text)
            word_stripped = word.strip()
            text = new_text

            # 如果知识库中有答案，使用知识库的回答
            if knowledge_base_response:
                yield TextGenerationStreamResponse(
                    generated_text=knowledge_base_response,
                    token=Token(
                        id=0,
                        logprob=0,
                        text=knowledge_base_response,
                        special=False
                    )
                )
            # 如果知识库中没有答案，使用默认回答
            elif default_response:
                yield TextGenerationStreamResponse(
                    generated_text=default_response,
                    token=Token(
                        id=0,
                        logprob=0,
                        text=default_response,
                        special=False
                    )
                )
            # 如果是模型生成的文本，正常处理
            else:
                yield TextGenerationStreamResponse(
                    generated_text=text,
                    token=Token(
                        id=0,
                        logprob=0,
                        text=word,
                        special=word_stripped.startswith('<|') and word_stripped.endswith('|>'),
                    )
                )
