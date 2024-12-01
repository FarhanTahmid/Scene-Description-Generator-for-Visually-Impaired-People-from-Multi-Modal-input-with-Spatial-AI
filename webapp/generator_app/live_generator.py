import cv2
from PIL import Image
import torch

class LiveFeedDescriptionGenerator():
    
    def __init__(self, processor, model):
        self.processor = processor
        self.model = model
     
    def generate_scene_description(self,frame):
        """Generate a scene description for a given frame."""
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert frame to PIL format
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model.generate(**inputs)
        caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        return caption
    
   
    