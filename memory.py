# -*- coding: utf-8 -*-
# @Time    : 2023/3/21 13:04
# @Author  : BarryWang
# @FileName: memory.py
# @Github  : https://github.com/BarryWangQwQ
import time
import uuid

from txtai.embeddings import Embeddings


def print_log(text: str):
    print(
        '[{0}] {1}'.format(
            time.strftime(
                '%Y-%m-%d %H:%M:%S',
                time.localtime(time.time())
            ), text
        )
    )


class Dialogue:
    user_content: str
    assistant_content: str

    def __init__(self, user_content: str, assistant_content: str):
        self.user_content = user_content
        self.assistant_content = assistant_content

    def raw(self):
        return [
            {'role': 'user', 'content': self.user_content},
            {'role': 'assistant', 'content': self.assistant_content}
        ]


class MemoryBlocks:
    embeddings: Embeddings
    model: str
    length: int

    def __init__(
            self,
            length: int = 5,
            model: str = 'sentence-transformers/distiluse-base-multilingual-cased-v2'
    ):
        self.model = model
        self.length = length
        self.embeddings = Embeddings(
            {
                'path': self.model,
                'content': True
            }
        )
        print_log('Analog memory block already loaded.')

    def upsert(self, dialogue_list):
        self.embeddings.upsert(
            (
                str(uuid.uuid4()), {'text': dialogue.user_content, 'raw': dialogue.raw()}, None
            ) for dialogue in dialogue_list
        )

    def search(self, question: str) -> list:
        neighborhoods = []
        results = self.embeddings.search(
            "SELECT text, score, raw FROM txtai WHERE similar('{0}') limit {1}".format(question, self.length)
        )
        for r in results:
            neighborhoods += eval(r['raw'])
        return list(reversed(neighborhoods))

    def reset(self):
        self.embeddings.close()
        self.embeddings = Embeddings(
            {
                'path': self.model,
                'content': True
            }
        )

    def save(self, output_path):
        self.embeddings.save(output_path)

    def load(self, load_path):
        self.embeddings.load(load_path)

    def info(self):
        return self.embeddings.search(
            "SELECT raw FROM txtai limit 9999999999"
        )
