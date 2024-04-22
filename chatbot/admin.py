from django.contrib import admin
from .models import AssessmentQuestion,AssessmentHistory,Document,Subject,AssessType,Question,Answer

admin.site.register(Question)
admin.site.register(Answer)


admin.site.register(AssessmentQuestion)
admin.site.register(AssessmentHistory)
admin.site.register(Subject)
admin.site.register(AssessType)

admin.site.register(Document)