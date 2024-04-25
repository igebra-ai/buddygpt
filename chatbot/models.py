from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
    
class AssessmentQuestion(models.Model):
    question = models.TextField()
    options = models.JSONField()
    answer = models.CharField(max_length=255)

    def __str__(self):
        return self.question
    
class Subjects(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    subjects = models.CharField(max_length=100)

    def __str__(self):
        return self.subjects  
    
class AssessmentType(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    types = models.CharField(max_length=100)

    def __str__(self):
        return self.types  

class AssessmentHistory(models.Model):
    assessment_id = models.CharField(max_length=255, primary_key=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    score = models.IntegerField()
    max_score = models.IntegerField()
    result_details = models.JSONField()
    # Add a datetime field to store the timestamp of the assessment
    date_taken = models.DateTimeField(auto_now_add=True, null=False)
    subjects = models.ForeignKey('Subjects', on_delete=models.CASCADE , related_name='assessments')
    assess_type = models.ForeignKey(AssessmentType, on_delete=models.CASCADE)  # Link to AssessType
    
    def __str__(self):
        return f"Assessment History ID: {self.assessment_id} - User: {self.user.username}"


class Document(models.Model):
    name = models.CharField(max_length=255)
    file = models.FileField(upload_to='documents/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name
    
    
from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    address_line_1 = models.CharField(max_length=255, blank=True)
    address_line_2 = models.CharField(max_length=255, blank=True)
    city = models.CharField(max_length=100, blank=True)
    state = models.CharField(max_length=100, blank=True)
    country = models.CharField(max_length=100, blank=True)
    contact_no = models.CharField(max_length=15, blank=True)
    profile_picture = models.ImageField(upload_to='profile_pics/', blank=True, null=True)

    def __str__(self):
        return f"{self.user.username}'s Profile " 

@receiver(post_save, sender=User)
def create_or_update_user_profile(sender, instance, created, **kwargs):
    if created:
        Profile.objects.create(user=instance)
    instance.profile.save()
