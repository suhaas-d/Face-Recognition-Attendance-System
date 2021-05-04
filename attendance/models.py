from django.db import models
from django.contrib.postgres.fields import ArrayField

# Create your models here.
class student(models.Model) :
    name = models.CharField(max_length = 100)
    face_encoding = ArrayField(models.IntegerField(), size = 128, )
    class_section = models.CharField(max_length = 5)
