from django.db import models

# Create your models here.

class face_detected(models.Model) :
    photo = models.ImageField(upload_to='detected')