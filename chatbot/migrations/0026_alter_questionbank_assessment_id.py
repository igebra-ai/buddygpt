# Generated by Django 5.0.6 on 2024-07-20 06:02

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("chatbot", "0025_questionbank_assessment_id"),
    ]

    operations = [
        migrations.AlterField(
            model_name="questionbank",
            name="assessment_id",
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
    ]
