# Generated by Django 4.1.13 on 2024-04-25 16:10

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("chatbot", "0007_profile"),
    ]

    operations = [
        migrations.AddField(
            model_name="assessmenthistory",
            name="subject",
            field=models.CharField(default="", max_length=255),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name="assessmenthistory",
            name="topic",
            field=models.CharField(default="", max_length=255),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name="assessmenthistory",
            name="type",
            field=models.CharField(default="", max_length=255),
            preserve_default=False,
        ),
    ]
