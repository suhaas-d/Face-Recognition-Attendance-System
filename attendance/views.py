from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.auth.models import User, auth
from django.contrib import messages
from django.core.files.storage import FileSystemStorage
import sys
from django.contrib import messages
from django.http import HttpResponse
sys.path.append("/home/suhaas/learndj/projects/friday/upload")
from try2 import *
# Create your views here.

def index(request):
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']

        user = auth.authenticate(username = username , password = password)
        if user is not None:
            auth.login(request, user)
            return redirect('upload/send')
        else: 
            messages.info(request, 'Invalid Credentials')
            return redirect('index')
    else:
        return render(request,'index.html')

username = ""        


def register(request):
    if request.method == 'POST':
        first_name = request.POST['first_name']
        last_name = request.POST['last_name']
        global username
        username = request.POST['username']
        email = request.POST['email']
        password1 = request.POST['password1']
        password2 = request.POST['password2']
        if password1==password2:
            if User.objects.filter(username = username).exists():
                messages.info(request, 'Username Taken')
                return redirect('register')
            elif User.objects.filter(email = email ).exists():
                messages.info(request, 'Email Taken')
                return redirect('register')
            else:
                user = User.objects.create_user(username = username, password = password1, email = email, first_name = first_name, last_name = last_name)
                user.save()
                print('user created')
                return redirect('register2')
        else :
            messages.info(request, 'Passwords Not Matching')
            return redirect('register')

    else:
        return render(request,'register.html')

def register2(request):
    context = {}
    if request.method == "POST":
        file_names = []
        images = request.FILES.getlist('images')
        print(len(images))
        for afile in images:
            file_name = ""
            fs = FileSystemStorage()
            name = fs.save(afile.name, afile)
            context['url'] = fs.url(name)
            test_img = './media/'+afile.name
            file_name = afile.name
            file_names.append(file_name)
            
        print(file_names)
        individual_encodings(file_names, username)   
        print('pickle file of students face encodings is saved')
        return redirect('index')
    
    else:
        messages.info(request, 'Upload the image as "Student Name".JPG')
        return render(request,'register2.html')