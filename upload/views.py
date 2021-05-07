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
import xlsxwriter
import datetime
import io
import os
# Create your views here.
students_names_encodings = {}
class_name = ""
msgs = []
val=[]
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
        global msgs
        global val
        msgs , val= get_attendance(test_img, class_name)
        os.remove(test_img)
        return redirect(download)
    else:
        messages.info(request, 'Upload the image as "your-classname/ same as your username".JPG')
        return render(request, 'send.html', context)

def download(request):
    if request.method=="POST":
        return render(request, 'download.html', {'msgs' : msgs, 'name': class_name})
    else:
        return render(request, 'download.html', {'name': class_name})

def download_excel(request):
    #response = HttpResponse( content_type = 'application/ms-excel' )
    output = io.BytesIO()
    

    workbook = xlsxwriter.Workbook(output, {'in_memory': True})
    now=datetime.datetime.now()
    worksheet = workbook.add_worksheet(class_name)
    worksheet.set_column(0, 1, 20)  # Width of columns A:A set to 30.
    cell_format = workbook.add_format({'bold': True, 'italic': False})
    row = 0

    for col, data in enumerate(val):
        if row!=0 or row != 2:
            worksheet.write_column(row, col, data)
        else:
            worksheet.write_column(row, col, data, cell_format)

    workbook.close()
    # construct response
    output.seek(0)
    response = HttpResponse(output.read(), content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    response['Content-Disposition'] = "attachment; filename=attendance_sheet.xlsx"
    #response['Content-Disposition'] = 'attachment; filename= attendance_sheet.xls'
    output.close()
    return response