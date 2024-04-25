from django.contrib import admin
from .models import AssessmentQuestion,AssessmentHistory,Document,Profile,Subjects,AssessmentType


admin.site.register(Subjects)
admin.site.register(AssessmentType)

admin.site.register(AssessmentQuestion)
admin.site.register(AssessmentHistory)
admin.site.register(Document)
admin.site.register(Profile)
