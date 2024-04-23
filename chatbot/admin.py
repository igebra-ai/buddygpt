from django.contrib import admin
from .models import AssessmentQuestion,AssessmentHistory,Document,Profile,Subject,AssessType

admin.site.register(AssessmentQuestion)
admin.site.register(AssessmentHistory)
admin.site.register(Document)
admin.site.register(Profile)
admin.site.register(Subject)
admin.site.register(AssessType)