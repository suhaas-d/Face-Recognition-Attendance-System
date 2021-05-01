from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.auth.models import User, auth
from django.views.generic import TemplateView
from django.core.files.storage import FileSystemStorage
import sys
sys.path.append("/home/suhaas/learndj/projects/friday/upload")
from brain import *

# Create your views here.

def send(request):
    context = {}
    if request.method=="POST":
        uploaded_file = request.FILES["upl"]
        fs = FileSystemStorage()
        name = fs.save(uploaded_file.name, uploaded_file)
        context['url'] = fs.url(name)
        test_img = './media/'+uploaded_file.name
        crop_faces_test(test_img)
    return render(request, 'send.html', context)

