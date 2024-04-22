from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
    
class AssessmentQuestion(models.Model):
    question = models.TextField()
    options = models.JSONField()
    answer = models.CharField(max_length=255)

    def __str__(self):
        return self.question
 
class Subject(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    subject = models.CharField(max_length=100)

    def __str__(self):
        return self.subject  
    
class AssessType(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    type = models.CharField(max_length=100)

    def __str__(self):
        return self.type      

class AssessmentHistory(models.Model):
    assessment_id = models.CharField(max_length=255, primary_key=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    score = models.IntegerField()
    max_score = models.IntegerField()
    result_details = models.JSONField()
    # Add a datetime field to store the timestamp of the assessment
    date_taken = models.DateTimeField(auto_now_add=True, null=False)
    subjects = models.ForeignKey('Subject', on_delete=models.CASCADE , related_name='assessments')
    assess_type = models.ForeignKey(AssessType, on_delete=models.CASCADE)  # Link to AssessType
    
     
    def __str__(self):
        return f"Assessment History ID: {self.assessment_id} - User: {self.user.username}"

class Document(models.Model):
    name = models.CharField(max_length=255)
    file = models.FileField(upload_to='documents/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name
    
class Question(models.Model):
    text = models.TextField()

    def __str__(self):
        return self.text

class Answer(models.Model):
    question = models.ForeignKey(Question, related_name='answers', on_delete=models.CASCADE)
    text = models.TextField()
    redirect_url = models.URLField(blank=True, null=True)  # Redirect to more information

    def __str__(self):
        return self.text
