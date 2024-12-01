# consumers.py
import cv2
import numpy as np
from channels.generic.websocket import AsyncWebsocketConsumer
import json
from gtts import gTTS
import os
import asyncio
from .live_generator import LiveFeedDescriptionGenerator
from transformers import BlipProcessor, BlipForConditionalGeneration

class VideoFeedConsumer(AsyncWebsocketConsumer):
    
    live_generator = LiveFeedDescriptionGenerator(
        processor=BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base"),
        model=BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    )

    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data=None, bytes_data=None):
        if bytes_data:
            # Convert the bytes data to a numpy array
            nparr = np.frombuffer(bytes_data, np.uint8)
            # Decode the numpy array to get the frame
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Generate caption using the pre-trained model
            generated_caption = VideoFeedConsumer.live_generator.generate_scene_description(frame)

            # Generate and save the audio using gTTS
            tts = gTTS(text=generated_caption, lang='en')
            audio_file_path = "media/generated_audio.mp3"
            tts.save(audio_file_path)

            # Create the full URL for accessing the audio file
            audio_url = f'http://127.0.0.1:8000/media/generated_audio.mp3'

            # Send the caption and audio URL back to the client
            await self.send(text_data=json.dumps({
                'caption': generated_caption,
                'audio_url': audio_url,
            }))

            # Add a delay to prevent overwhelming the system
            await asyncio.sleep(3)
