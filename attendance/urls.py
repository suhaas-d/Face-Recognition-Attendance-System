
from django.urls import path
from . import views

urlpatterns = [
    path('',views.index,name = 'index'),
    path('register',views.register, name = 'register'),
    path('register2',views.register2, name = 'register2'),
    path('index',views.index,name = 'index')
]
