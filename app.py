# -*- coding: utf-8 -*-
# @Time    : 2023/3/19 22:36
# @Author  : BarryWang
# @FileName: CharacterGPT.py
# @Github  : https://github.com/BarryWangQwQ

import os
from pprint import pprint

import yaml
import time
import openai
import gradio as gr
import warnings
from memory import Dialogue, MemoryBlocks

warnings.filterwarnings("ignore")

with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


def print_log(text: str):
    print(
        '[{0}] {1}'.format(
            time.strftime(
                '%Y-%m-%d %H:%M:%S',
                time.localtime(time.time())
            ), text
        )
    )


def print_character_config():
    print_log('Loading character config...')
    pprint(config)


print_character_config()

mb = MemoryBlocks(
    length=config['traceback_length'],
    model=config['vector_space_model']
)

openai.api_key = config['openai_key']

character_name = config['character']
title = f"Chat - {character_name}"

prompt_default = ""
frame_value = []

if os.path.exists(character_name):
    mb.load(character_name)
    with open(os.path.join(character_name, 'prompts'), 'r', encoding='utf-8') as f:
        prompt_default = f.read()
    for r in mb.info():
        raw = eval(r['raw'])
        frame_value.append([raw[0]['content'], raw[1]['content']])

messages = [{"role": "system", "content": prompt_default}]

# def overflow_token(messages_list, string):
#     total_length = len(string)
#     for messages_dict in messages_list:
#         for k, v in messages_dict.items():
#             total_length += len(str(k)) + len(str(v))
#     if total_length <= 3000:
#         return False
#     return True


with gr.Blocks(title=title) as demo:
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(lines=18, label="引导参数", value=prompt_default,
                                placeholder='请用文字描述一下AI需要扮演的角色~')
            system_log = gr.Textbox(label="系统消息", placeholder='', interactive=False)
            reset = gr.Button("重置模型")
            clear = gr.Button("清空聊天")
        with gr.Column():
            chatbot = gr.Chatbot().style(height=520)
            msg = gr.Textbox(label="聊天")
    with gr.Column():
        dataframe = gr.Dataframe(
            headers=["用户", "角色"],
            datatype=["str", "str"],
            col_count=(2, "fixed"),
            max_rows=9999999999,  # -> +INF
            type='array',
            value=frame_value,
            interactive=True
        )
        with gr.Row():
            update = gr.Button("更新")
            save = gr.Button("导出")


    def user(prompts, user_message, history):
        # print(history + [[user_message, None]])
        print("prompt:", prompts)
        global messages
        # try:
        #     print("prompt:", prompts[0: prompts.find("###EXAMPLE_START###")])
        #     print("examples:", prompts[prompts.find("###EXAMPLE_START###") + 19: prompts.find('###EXAMPLE_END###')])
        # except (Exception, BaseException):
        #     pass
        # try:
        #     examples = eval(
        #         "[" + prompt[prompt.find("###EXAMPLE_START###") + 19: prompt.find('###EXAMPLE_END###')] + "]")
        # except (Exception, BaseException):
        #     examples = []
        # if not examples:
        #     messages[0]["content"] = prompts
        # else:
        #     messages[0]["content"] = prompts[0: prompts.find("###EXAMPLE_START###")]
        #     while overflow_token(messages, user_message):
        #         messages.pop(1)
        #     for example in examples:
        #         if example not in messages:
        #             messages.insert(1, example)
        messages = [{"role": "system", "content": ""}]
        messages[0]["content"] = prompts
        return "", history + [[user_message, None]]


    def character(history):
        global messages
        neighborhoods = mb.search(history[-1][0])
        messages += neighborhoods
        messages.append({"role": "user", "content": history[-1][0]})

        print('messages:', messages)
        print('user:', {"role": "user", "content": history[-1][0]})
        print('neighborhoods:', neighborhoods)

        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        reply = chat.choices[0].message.content
        print('assistant:', {"role": "assistant", "content": reply})

        mb.upsert([Dialogue(history[-1][0], reply)])

        # messages.append({"role": "assistant", "content": reply})

        history[-1][1] = reply
        return history


    def reset_model():
        global messages
        messages = [{"role": "system", "content": prompt_default}]
        mb.reset()
        print_log('Reset Model.')
        return [[None, "已重置模型"]]


    def update_example(data_frame):
        temp = []
        database = []
        dialogues = []
        # print('data_frame', data_frame)

        for dialogue in data_frame:
            if dialogue[0] != '' and dialogue[1] != '':
                temp.append((dialogue[0], dialogue[1]))

        for r in mb.info():
            raw = eval(r['raw'])
            database.append((raw[0]['content'], raw[1]['content']))

        upsert_list = list(
            set(temp).difference(set(database))
        )

        for dialogue in upsert_list:
            dialogues.append(Dialogue(dialogue[0], dialogue[1]))

        if dialogues:
            mb.upsert(dialogues)

        # print('upsert_list', upsert_list)

        database = []
        for r in mb.info():
            raw = eval(r['raw'])
            database.append([raw[0]['content'], raw[1]['content']])

        # print('database', database)

        if not database:
            database = [['', '']]

        print_log('Update memory block.')

        return '[{0}] 已更新记忆区块'.format(
            time.strftime(
                '%Y-%m-%d %H:%M:%S',
                time.localtime(time.time())
            )
        ), database


    def save_model(prompts):
        mb.save(character_name)

        with open(os.path.join(character_name, 'prompts'), 'w+', encoding='utf-8') as f:
            f.write(prompts)

        print_log('Export character.')

        return '[{0}] 已导出角色记忆模型'.format(
            time.strftime(
                '%Y-%m-%d %H:%M:%S',
                time.localtime(time.time())
            )
        )


    msg.submit(
        user, [prompt, msg, chatbot], [msg, chatbot],
        queue=False,
        api_name='chat'
    ).then(
        character, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)
    reset.click(reset_model, None, chatbot, queue=False, api_name='reset')
    update.click(update_example, dataframe, [system_log, dataframe], queue=False, api_name='update')
    save.click(save_model, prompt, system_log, queue=False, api_name='save')


def launch_demo():
    print_log('Launching DEMO..')
    demo.launch(show_api=config['character'])


if __name__ == "__main__":
    launch_demo()
