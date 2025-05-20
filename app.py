import gradio as gr
from fastai.learner import load_learner
from PIL import Image
import numpy as np

def get_label(x):
    return [x.parent.name]

models = {
    "ResNet50_multicategoryblock_v3": load_learner("model_multicategoryblock_resnet50_v3.pkl"),
    "ResNet18_multicategoryblock_v3": load_learner("model_multicategoryblock_resnet18_v3.pkl")
}

def predict(image, model_name):
    learn = models[model_name]
    pred, _, probs = learn.predict(image)

    probs_array = np.array(probs)
    max_prob = np.max(probs_array)
    min_prob = np.min(probs_array)
    spread = max_prob - min_prob

    if spread < 0.2:  
        return {"Error": "This doesn't look like a brain MR image."}

    if max_prob < 0.7:
        return {"Error": "Low confidence. The image might not be a valid brain MR."}

    return {cls: float(prob) for cls, prob in zip(learn.dls.vocab, probs)}

with gr.Blocks() as demo:
    gr.Markdown("## Brain Tumor Classification with Multiple Models")
    gr.Markdown("Upload a brain MR image and choose a model to classify it into tumor types or normal.")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload MR Image")
        model_input = gr.Dropdown(choices=list(models.keys()), label="Select Model")

    output_label = gr.Label(num_top_classes=4)

    predict_button = gr.Button("Classify", variant="primary")
    predict_button.click(fn=predict, inputs=[image_input, model_input], outputs=output_label)

    gr.Markdown("[📖 Read the Blog from Medium](https://medium.com/@busraemir55/brain-tumor-classification-with-deep-learning-and-fastai-4ee60eaf19b6) &nbsp;&nbsp;&nbsp;\
                 [💻 View the Training Code from GitHub](https://github.com/busra-emir/brain_tumor_classification) &nbsp;&nbsp;&nbsp;\
                 [📁 View the Dataset Used for Training](https://www.kaggle.com/datasets/thomasdubail/brain-tumors-256x256)"
               )

if __name__ == "__main__":
    demo.launch()
    
