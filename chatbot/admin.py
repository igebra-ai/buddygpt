from django.contrib import admin
from .models import AssessmentQuestion,AssessmentHistory,Document,Profile,Sub,Assess


admin.site.register(Sub)
admin.site.register(Assess)


admin.site.register(AssessmentQuestion)
admin.site.register(AssessmentHistory)
admin.site.register(Document)
admin.site.register(Profile)
