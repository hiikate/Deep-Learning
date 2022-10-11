# %%
! pip install -q gradio

# %%
#\export
from fastai.vision.all import *
import gradio as gr

# %%
learn = load_learner('model.pkl')

# %%
im = PILImage.create('broken_car.jpg')

# %%
learn.predict(im)

# %%
#\export
categories = ('Broken', 'Normal')

def classifty_image(img):
    pred, idx, prob = learn.predict(img)
    return dict(zip(categories, map(float,prob)))

# %%
classify_image(im)

# %%
#\export
from doctest import Example


image = gr.Image()
label = gr.Label()
examples = ['Broken_car.jpg','normal_car.jpg']

demo = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples, allow_flagging = "never", thumbnail=thumbnail)
demo.launch(inline=False)


