import torch
import gradio as gr
import spaces
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import re
from polyglot.detect import Detector

HF_TOKEN = os.environ.get("HF_TOKEN", None)
MODEL = "LLaMAX/LLaMAX3-8B-Alpaca"
RELATIVE_MODEL="LLaMAX/LLaMAX3-8B"

TITLE = "<h1><center>LLaMAX Translator</center></h1>"


model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float16,
        device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL)


def lang_detector(text):
    min_chars = 5
    if len(text) < min_chars:
        return "Input text too short"
    try:
        detector = Detector(text).language
        lang_info = str(detector)
        code = re.search(r"name: (\w+)", lang_info).group(1)
        return code
    except Exception as e:
        return f"ERROR：{str(e)}"

def Prompt_template(inst, prompt, query, src_language, trg_language):
    inst = inst.format(src_language=src_language, trg_language=trg_language)
    instruction = f"`{inst}`"
    prompt = (
        f'{prompt}'
        f'### Instruction:\n{instruction}\n'
        f'### Input:\n{query}\n### Response:'
    )
    return prompt

# Unfinished
def chunk_text():
    pass
    
@spaces.GPU(duration=60)
def translate(
    source_text: str, 
    source_lang: str,
    target_lang: str,
    inst: str, 
    prompt: str, 
    max_length: int,
    temperature: float,
    top_p: float,
    rp: float):
    
    print(f'Text is - {source_text}')
    
    prompt = Prompt_template(inst, prompt, source_text, source_lang, target_lang)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        
    generate_kwargs = dict(
        input_ids=input_ids,
        max_length=max_length, 
        do_sample=True, 
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=rp,    
    )

    outputs = model.generate(**generate_kwargs)
    
    resp = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    yield resp[len(prompt):]

CSS = """
    h1 {
        text-align: center;
        display: block;
        height: 10vh;
        align-content: center;
        font-family: Arial, Helvetica, sans-serif;
    }
    footer {
        visibility: hidden;
    }
    font-family: Arial, Helvetica, sans-serif;
"""

LICENSE = """
Model: <a href="https://huggingface.co/LLaMAX/LLaMAX3-8B-Alpaca">LLaMAX3-8B-Alpaca</a>
"""

LANG_LIST = ['Akrikaans', 'Amharic', 'Arabic', 'Armenian', 'Assamese', 'Asturian', 'Azerbaijani', \
             'Belarusian', 'Bengali', 'Bosnian', 'Bulgarian', 'Burmese', \
             'Catalan', 'Cebuano', 'Simplified Chinese', 'Traditional Chinese', 'Croatian', 'Czech', \
             'Danish', 'Dutch', 'English', 'Estonian', 'Filipino', 'Finnish', 'French', 'Fulah', \
             'Galician', 'Ganda', 'Georgian', 'German', 'Greek', 'Gujarati', \
             'Hausa', 'Hebrew', 'Hindi', 'Hungarian', \
             'Icelandic', 'Igbo', 'Indonesian', 'Irish', 'Italian', \
             'Japanese', 'Javanese', \
             'Kabuverdianu', 'Kamba', 'Kannada', 'Kazakh', 'Khmer', 'Korean', 'Kyrgyz', \
             'Lao', 'Latvian', 'Lingala', 'Lithuanian', 'Luo', 'Luxembourgish', \
             'Macedonian', 'Malay', 'Malayalam', 'Maltese', 'Maori', 'Marathi', 'Mongolian', \
             'Nepali', 'Northern', 'Norwegian', 'Nyanja', \
             'Occitan', 'Oriya', 'Oromo', \
             'Pashto', 'Persian', 'Polish', 'Portuguese', 'Punjabi', \
             'Romanian', 'Russian', \
             'Serbian', 'Shona', 'Sindhi', 'Slovak', 'Slovenian', 'Somali', 'Sorani', 'Spanish', 'Swahili', 'Swedish', \
             'Tajik', 'Tamil', 'Telugu', 'Thai', 'Turkish', \
             'Ukrainian', 'Umbundu', 'Urdu', 'Uzbek', \
             'Vietnamese', 'Welsh', 'Wolof', 'Xhosa', 'Yoruba', 'Zulu']

chatbot = gr.Chatbot(height=600)

with gr.Blocks(theme="soft", css=CSS) as demo:
    gr.Markdown(TITLE)
    with gr.Row():
        with gr.Column(scale=4):
            source_text = gr.Textbox(
                label="Văn bản gốc",
                value="LLaMAX is a language model with powerful multilingual capabilities without loss instruction-following capabilities. "+\
                "LLaMAX supports translation between more than 100 languages, "+\
                "surpassing the performance of similarly scaled LLMs.",
                lines=10,
            )
            output_text = gr.Textbox(
                label="Văn bản đã được dịch",
                lines=10,
                show_copy_button=True,
            )

        with gr.Column(scale=1):
            source_lang = gr.Dropdown(
                label="Ngôn ngữ nguồn",
                value="English",
                choices=LANG_LIST,
            )
            target_lang = gr.Dropdown(
                label="Ngôn ngữ đích",
                value="Vietnamese",
                choices=LANG_LIST,
            )
            max_length = gr.Slider(
                label="Độ dài tối đa",
                minimum=512,
                maximum=8192,
                value=4000,
                step=8,
            )
            temperature = gr.Slider(
                label="Temperature",
                minimum=0,
                maximum=1,
                value=0.3,
                step=0.1,
            )
            top_p = gr.Slider(
                label="top_p",
                minimum=0.0,
                maximum=1.0,
                step=0.1,
                value=1.0,
            )
            rp = gr.Slider(
                label="Repetition penalty",
                minimum=1.0,
                maximum=2.0,
                step=0.1,
                value=1.2,
            )
            with gr.Accordion("Tùy chọn nâng cao", open=False):
                inst = gr.Textbox(
                    label="Instruction",
                    value="Translate the following sentences from {src_language} to {trg_language}.",
                    lines=3,
                )
                prompt = gr.Textbox(
                    label="Prompt",
                    # Prompt 1
                    #value="""Below is an instruction that describes a task, paired with an input that provides further context.
#Write a response that appropriately completes the request.
### Instruction:
#{instruction}
### Input:
#{query}
### Response:""",#
                    # Prompt 2
                    value="""Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that ensuring accuracy and maintaining the tone and style of the original text.
### Instruction:
{instruction}
### Input:
{query}
### Response:""",
                    lines=8,
                )
                
    with gr.Row():
        submit = gr.Button(value="Submit")
        clear = gr.ClearButton([source_text, output_text])
    gr.Markdown(LICENSE)
    
    #source_text.change(lang_detector, source_text, source_lang)
    submit.click(fn=translate, inputs=[source_text, source_lang, target_lang, inst, prompt, max_length, temperature, top_p, rp], outputs=[output_text])


if __name__ == "__main__":
    demo.launch()
