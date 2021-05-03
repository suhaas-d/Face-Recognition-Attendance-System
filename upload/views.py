from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.auth.models import User, auth
from django.views.generic import TemplateView
from django.core.files.storage import FileSystemStorage
import sys
from django.contrib import messages
from django.http import HttpResponse
sys.path.append("/home/suhaas/learndj/projects/friday/upload")
from brain import *

# Create your views here.
students_names_encodings = {}
class_name = ""
msg_dict = {}
def send(request):
    context = {}
    if request.method=="POST":
        uploaded_file = request.FILES["upl"]
        fs = FileSystemStorage()
        name = fs.save(uploaded_file.name, uploaded_file)
        context['url'] = fs.url(name)
        test_img = './media/'+uploaded_file.name
        global class_name
        class_name = uploaded_file.name[:-4]
        crop_for_attendance(test_img, class_name)
        global msg_dict
        msg_dict = get_attendance(class_name)
        #os.remove(test_img)
        return redirect(download)
    else:
        messages.info(request, 'Upload the image as "your-classname/ same as your username".JPG')
        return render(request, 'send.html', context)

def download(request):
    if request.method=="POST":
        diction = {}
        diction = generate_encodings(class_name)
        print(class_name)
 
        return render(request, 'download.html', {'msgs' : msg_dict, 'name': class_name})
    else:
        return render(request, 'download.html', {'name': class_name})
