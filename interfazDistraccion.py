import torch
import timm
import torch.nn as nn
import gradio as gr
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F



#####################################
# We define and initialize the model
#####################################

#We define the class for the model
class DistractionModel(nn.Module):
    def __init__(self, num_classes):
        super(DistractionModel, self).__init__()
        # Load pre-trained base model
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        enet_out_size = self.base_model.classifier.in_features
        # Modify classifier
        self.base_model.classifier = nn.Sequential(
            nn.Linear(enet_out_size, num_classes)
        )

    def forward(self, x):
        # Forward propagation
        x = self.base_model(x)
        return x


model_path = "C:/Users/usuario/Desktop/modeloMezcla19-05.pth"
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

# Extract the number of classes
num_classes = checkpoint['base_model.classifier.0.weight'].shape[0]

# Create the model with out number of classes
model = DistractionModel(num_classes=num_classes)

# Load the trained model's weights
model.load_state_dict(checkpoint)
model.eval()



#####################################
#Define the functions that will be used
#####################################

# Make the predicting function
def predict(imagen):
    # Make and apply the needed image transformations
    imagen_pil = Image.fromarray(imagen.astype('uint8'))
    transform = transforms.Compose([
        transforms.Resize((128, 128)), 
        transforms.ToTensor(),  
    ])
    imagen_transformada = transform(imagen_pil)

    # Make the prediction 
    salida = model(imagen_transformada.unsqueeze(0))
    
    # Apply softmax to obtain the probabilities
    probabilidades = F.softmax(salida, dim=1)

    # We obtain the labels
    label = ['Safe driving','Reckless driving'] 
    output_text = "Predicted class: "
    max_index = probabilidades.argmax()
    output_text += f"{label[max_index]}"

    return output_text

# Define a function to load and pre-process the image 
def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return image, transform(image).unsqueeze(0)



#####################################
# Configure the interface and launch it
#####################################

interface = gr.Interface(
    fn=predict,
    theme = gr.themes.Soft(primary_hue="indigo").set(
    loader_color="#3300FF",
    slider_color="#3300FF",
),
    allow_flagging="never",
    inputs="image",
    outputs="text",
    title="Safe driving classification",
    description="Load an image and it'll be classified"
)

# Launch the interface
interface.launch()
