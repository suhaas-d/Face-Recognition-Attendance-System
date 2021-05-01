# Generated by Django 3.1.7 on 2021-04-24 11:48

import django.contrib.postgres.fields
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='student',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('face_encoding', django.contrib.postgres.fields.ArrayField(base_field=models.IntegerField(), size=128)),
                ('class_section', models.CharField(max_length=5)),
            ],
        ),
    ]