# Generated by Django 4.1.13 on 2024-04-26 10:24

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('chatbot', '0015_answer_question_delete_assess_answer_question'),
    ]

    operations = [
        migrations.AlterField(
            model_name='answer',
            name='question',
            field=models.TextField(max_length=1500),
        ),
    ]