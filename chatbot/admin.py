from django.contrib import admin
from .models import AssessmentQuestion,AssessmentHistory,Document

admin.site.register(AssessmentQuestion)
admin.site.register(AssessmentHistory)
admin.site.register(Document)