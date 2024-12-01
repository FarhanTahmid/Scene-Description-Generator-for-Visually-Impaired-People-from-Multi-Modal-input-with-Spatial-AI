from django.urls import path
from . import views

app_name="generator_app"

urlpatterns = [
    path('', views.landingPage, name='landingPage'),
    path('generator/',views.generatePage,name='generator'),
    path('generate_description/',views.generate_description,name="generate_description"),
    path('generate_description_by_scene/',views.generateDescriptionByScene,name="generate_description_by_scene"),
    path('generate_audio/',views.generate_audio,name="generate_audio"),
    
]