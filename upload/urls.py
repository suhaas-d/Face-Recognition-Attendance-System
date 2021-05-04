from django.urls import path
from . import views

urlpatterns = [
    path('send',views.send,name = 'send'),
    path('download', views.download, name = 'download')
]
