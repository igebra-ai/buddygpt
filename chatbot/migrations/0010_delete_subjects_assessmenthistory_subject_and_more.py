# Generated by Django 4.1.13 on 2024-04-25 21:11

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("chatbot", "0009_subjects_remove_assessmenthistory_subject_and_more"),
    ]

    operations = [
        migrations.DeleteModel(
            name="subjects",
        ),
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