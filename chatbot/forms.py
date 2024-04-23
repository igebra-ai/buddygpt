from django import forms
from .models import Document

class DocumentForm(forms.ModelForm):
    class Meta:
        model = Document
        fields = ['name', 'file']

    def clean_file(self):
        file = self.cleaned_data['file']
        allowed_types = ['text/plain', 'application/pdf', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/csv']

        if file.content_type not in allowed_types:
            raise forms.ValidationError("File format not supported. Please upload a txt, pdf, doc, docx, or csv file.")

        return file


from .models import Profile

class ProfileForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = ['address_line_1', 'address_line_2', 'city', 'state', 'country', 'contact_no', 'profile_picture']
