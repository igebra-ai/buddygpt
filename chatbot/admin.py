from django.contrib import admin
from .models import AssessmentQuestion,AssessmentHistory,Document,Profile,AssessmentSubject,AssessmentTopic,AssessmentFormat,Question,Answer

admin.site.register(Question)
admin.site.register(Answer)



admin.site.register(AssessmentQuestion)
admin.site.register(AssessmentHistory)
admin.site.register(Document)
admin.site.register(Profile)
admin.site.register(AssessmentSubject)
admin.site.register(AssessmentTopic)
admin.site.register(AssessmentFormat)