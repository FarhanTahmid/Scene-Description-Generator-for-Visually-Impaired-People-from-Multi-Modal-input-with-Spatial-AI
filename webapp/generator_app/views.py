import PIL.Image
from django.shortcuts import render
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import api_view, permission_classes
from rest_framework import status
from django.http import JsonResponse,HttpResponse
from gtts import gTTS
from django.views.decorators.csrf import csrf_exempt
import os
import sys
import PIL
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from .generator import Generator
from .by_scene_generator import BySceneGenerator

# Create your views here.

def landingPage(request):
    return render(request,'index.html')

def generatePage(request):
    
    if request.method=="POST":
        file_to_process=request.FILES.get('input_file')
        print(file_to_process)
    return render(request,'generation_page.html')

@csrf_exempt
def generate_description(request):
    if request.method == "POST":
        file_object=request.FILES.get('file')
        if not file_object:
            return JsonResponse({"error": "No file uploaded"}, status=status.HTTP_400_BAD_REQUEST)
        else:
            with open(f"test_uploads/{file_object.name}", "wb+") as destination:
                for chunk in file_object.chunks():
                    destination.write(chunk)
                print("File Saved!")
            
            # get the file object stored in test_uploads folder to pass 
            # to the function that generates the description
            
            file_path = f"test_uploads/{file_object.name}"
            file_extension=Path(file_object.name).suffix
            
            img_extensions = ['.jpg', '.png', '.jpeg']
            vid_extensions = ['.mp4', '.avi']
            
            generator=Generator()

            
            if os.path.isfile(file_path):
                print("File exists")
                if file_extension.lower() in img_extensions:
                    print("its an image file here")                
                    caption=generator.gen_caption(file_path,as_string=True,by_scene=False)
                    return JsonResponse({"message":caption,"vidFile":False,"filename":file_object.name}, status=status.HTTP_200_OK)
 
                else:
                    if file_extension.lower() in vid_extensions:
                        caption=generator.gen_caption(file_path,as_string=True,by_scene=False)
                        print("its a video file here")                
                        caption=generator.gen_caption(file_path,as_string=True,by_scene=False)
                        return JsonResponse({"message":caption,"vidFile":True,"filename":file_object.name}, status=status.HTTP_200_OK)
                    else:
                        return JsonResponse({"error": f"The current model can not process {file_extension} files!"}, status=status.HTTP_400_BAD_REQUEST)

            else:
                print("File does not exist")
                return JsonResponse({"error": "No file found to pass for description generation"}, status=status.HTTP_400_BAD_REQUEST)

@csrf_exempt
def generateDescriptionByScene(request):
    if request.method == "POST":
        print("Generate Descriptions by scene!")
        file_object=request.FILES.get('file')
        if not file_object:
            return JsonResponse({"error": "No file uploaded"}, status=status.HTTP_400_BAD_REQUEST)
        else:
            with open(f"test_uploads/{file_object.name}", "wb+") as destination:
                for chunk in file_object.chunks():
                    destination.write(chunk)
                print("File Saved!")
            
            # get the file object stored in test_uploads folder to pass 
            # to the function that generates the description
            
            file_path = f"test_uploads/{file_object.name}"
            file_extension=Path(file_object.name).suffix
            
            img_extensions = ['.jpg', '.png', '.jpeg']
            vid_extensions = ['.mp4', '.avi']

            if file_extension in vid_extensions:
                by_scene_generator=BySceneGenerator()
                captions,scene_change_timecodes=by_scene_generator.generate_description(video_path=file_path,as_string=True,byScene=True)
                
                return JsonResponse({"captions":captions,"timecodes":scene_change_timecodes,"filename":file_object.name}, status=status.HTTP_200_OK)
            else:
                return JsonResponse({"message":"Can not generate descriptions by scene!"}, status=status.HTTP_200_OK)

            
            
        return JsonResponse({"message":"Generating Descriptions by scene!"}, status=status.HTTP_200_OK)
    else:
        return JsonResponse({"error": "No file found to pass for description generation"}, status=status.HTTP_400_BAD_REQUEST)


def generate_audio(request):
    captions=request.GET.get("captions","")
    filename=request.GET.get("file_name")
    
    if not captions:
        return JsonResponse({"error": "No captions provided for audio generation"}, status=400)
    try:
        # Generate audio from captions using gTTS
        tts = gTTS(text=captions, lang='en')
        audio_file_path = f"test_results/{filename}_generated_audio.mp3"
        tts.save(audio_file_path)

        # Serve the audio file as a response
        with open(audio_file_path, 'rb') as audio_file:
            response = HttpResponse(audio_file.read(), content_type="audio/mpeg")
            response['Content-Disposition'] = f'inline; filename="{audio_file_path}"'
            return response

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
    

