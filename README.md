import gradio as gr
import torch
from transformers import TextIteratorStreamer, AutoTokenizer
from transformers import TextIteratorStreamer
import spaces
from threading import Thread
# from peft import AutoPeftModelForCausalLM

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("djelia/bm-mistral-7b-v1")
model = AutoModelForCausalLM.from_pretrained("djelia/bm-mistral-7b-v1", device_map="auto")


# Task prompts from your notebook
TRANSLATION_BM_FR_PROMPT = """I ye kanbaara kɛla min bɛ kuma yɛlɛma ka bɔ "bambara" la ka taa "français" la.
I bɛna kuma sɔrɔ "bambara" la, i ka kan ka o yɛlɛma ka kɛ "français" ye.
I ka kan ka yɛlɛmali dɔrɔn di, ka to kunnafoni wɛrɛw ni kow ɲɛfɔli la.
"""

TRANSLATION_FR_BM_PROMPT = """I ye kanbaara kɛla min bɛ kuma yɛlɛma ka bɔ "français" la ka taa "bambara" la.
I bɛna kuma sɔrɔ "français" la, i ka kan ka o yɛlɛma ka kɛ "bambara" ye.
I ka kan ka yɛlɛmali dɔrɔn di, ka to kunnafoni wɛrɛw ni kow ɲɛfɔli la.
"""

SENTIMENT_PROMPT = """I ye sentiment classifier dɛmɛbaga ye min bɛ se ka Bamanankan kɔnɔ sentiment classification kɛ kosɛbɛ.
I ka baara ye ka mɔgɔw ka kuma sentiment dɔn ka bɔ ni sentiment ninnu na: positive, neutral, negative.
"""

TRANSCRIPTION_CORRECTION_PROMPT = """I ye Bambara sɛbɛnni kɔrɔsibaga ye min bɛ ASR Bambara sɛbɛnni ɲɛnabɔ. I ka baara ye ka Bambara sɛbɛnni fili minnu bɛ ASR la, olu yɛlɛma ka kɛ Bambara sɛbɛnni ɲuman ye, ka kɔrɔ bɛɛ to a cogo la. I ka kan ka fili suguyaw ninnu ɲɛnabɔ:

Daɲɛw Tilili: Tuma dɔw la, ASR bɛ daɲɛ kelen tila ka kɛ daɲɛ fitini caman ye. I ka kan ka olu fara ɲɔgɔn kan ka kɛ daɲɛ ɲuman kelen ye.
Daɲɛw Farali: Tuma dɔw la, daɲɛ fla bɛ fara ɲɔgɔn kan ka kɛ kelen ye. I ka kan ka olu tila ka Bambara sɛbɛnni cogo ɲuman bato.
Tubabukan Yɛlɛmali Fili: Tubabukan kumaw bɛ se ka yɛlɛma Bambara la ni kanfɔ suguya wɛrɛ ye min tɛ a ɲuman ye (misali la, "cette fois-ci" bɛ se ka kɛ "se ti fassi si" ye walima "à travers" bɛ kɛ "a taara were" ye). I ka kan ka olu sɛbɛn cogo ɲuman na.
Daɲɛw walima Sɛbɛndenw Tununi: Tuma dɔw la, daɲɛw walima sɛbɛnden kelenkelenna dɔw bɛ bɔ sɛbɛnni na.

Ni i bɛ Bambara kumakan dɔ ɲɛnabɔ, i ka kan ka fili suguyaw ninnu bɛɛ ɲɛnabɔ ka sɔrɔ ka kɛ Bambara sɛbɛnni ɲuman ye, nka ka kanfɔcogo bato walasa ka bɛn ni fɔcogo ye.
I ka labaaraw ka kan ka kɛ Bambara sɛbɛnni ɲɛnabɔlen ye, ni daɲɛw danw, tomi, ani daɲɛ sugandilen ɲumanw ye minnu bɛ bɛn ni kanfɔcogo ye. I ka jija ka kɔrɔ fɔlen to a cogo la ka sɔrɔ ka a ɲɛfɔ ka ɲɛ ani ka a kɛ sɛbɛnni ɲuman ye.
"""

alpaca_prompt = """Nin ye baara dɔ ɲɛfɔli ye, min bɛ donnafɛnw ni sigidaw fara ɲɔgɔn kan. I ka kan ka jaabi sɛbɛn min bɛ ɲinini dafa ka ɲɛ.

### ɲɛfɔli:
{}

### Donnafɛnw:
{}

### Jaabi:
{}
"""

def set_system_prompt(choice):
    prompts = {
        "Translation (BM-FR)": TRANSLATION_BM_FR_PROMPT,
        "Translation (FR-BM)": TRANSLATION_FR_BM_PROMPT,
        "Sentiment Analysis": SENTIMENT_PROMPT,
        "Transcription Correction": TRANSCRIPTION_CORRECTION_PROMPT,
        "Clear": ""
    }
    return prompts.get(choice, "")

@spaces.GPU()
def respond(message, history, system_message, max_tokens):
    formatted_prompt = alpaca_prompt.format(system_message, message, "")
    inputs = tokenizer([formatted_prompt], return_tensors="pt").to("cuda")

    streamer = TextIteratorStreamer(tokenizer)
    
    # Set up the generation parameters
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_tokens,
        use_cache = True
    )
    
    # Create and start generation thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Stream the response
    response = ""
    for new_text in streamer:
        response += new_text
        yield response.split("Jaabi:")[-1]

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.ChatInterface(
                respond,
                additional_inputs=[
                    gr.Textbox(label="System message", interactive=True),
                    gr.Slider(minimum=1, maximum=2048, value=300, step=1, label="Max new tokens")
                ]
            )
        
        with gr.Column(scale=1):
            task_buttons = gr.Radio(
                choices=["Translation (BM-FR)", 
                        "Translation (FR-BM)", 
                        "Sentiment Analysis", 
                        "Transcription Correction", 
                        "Clear"],
                label="Select Task",
                value=None
            )
    
    task_buttons.change(fn=set_system_prompt, inputs=[task_buttons], outputs=[chatbot.additional_inputs[0]])

demo.launch(share=True)

