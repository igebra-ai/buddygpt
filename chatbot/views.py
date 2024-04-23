from django.shortcuts import render, redirect
from .forms import DocumentForm
from django.http import JsonResponse
from openai import OpenAI
from django.contrib import auth
from django.contrib.auth.models import User
from .models import AssessmentQuestion, AssessmentHistory, Document
from django.shortcuts import render
from django.http import JsonResponse
import os
from dotenv import load_dotenv
import json
from django.db.models import Avg
from django.http import HttpResponse
from django.contrib import messages
from django.contrib.auth import authenticate, login, user_logged_in
from django_chatbot import settings
from django.core.mail import send_mail,EmailMessage
from django.contrib.sites.shortcuts import get_current_site
from django.template.loader import render_to_string
from django.utils.http import urlsafe_base64_decode, urlsafe_base64_encode
from django.utils.encoding import force_bytes
from django.contrib.auth import authenticate, login, logout
from . tokens import generate_token
from django.utils.encoding import force_bytes
try:
    from django.utils.encoding import force_text
except ImportError:
    from django.utils.encoding import force_str as force_text

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI()

def ask_openai(message):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a conversational chatbot."},
            {"role": "user", "content": message},
        ]
    )

    answer = response.choices[0].message.content.strip()
    return answer

# Create your views here.

def chatbot(request):
    if request.method == 'POST':
        message = request.POST.get('message')
        response = ask_openai(message)

        return JsonResponse({'message': message, 'response': response})
    return render(request, 'chatbot.html')

def chat(request):
    if not request.user.is_authenticated:
        # Redirect the user to the login page, or return a suitable response
        return redirect('signin') 
    if request.method == 'POST':
        message = request.POST.get('message')
        response = ask_openai(message)

        return JsonResponse({'message': message, 'response': response})
    return render(request, 'chat.html')

def signin(request):
    if request.method == 'POST':
        username = request.POST['username']
        pass1 = request.POST['pass1']
        
        user = authenticate(username=username, password=pass1)
        
        if user is not None:
            login(request, user)
            fname = user.first_name
            return redirect('dashboard')  # Redirect to dashboard URL
        else:
            messages.error(request, "Bad Credentials!")
            return redirect('signin')
            
    return render(request, "signin.html")


def register(request):
    if request.method == "POST":
        username = request.POST['username']
        fname = request.POST['fname']
        lname = request.POST['lname']
        email = request.POST['email']
        pass1 = request.POST['pass1']
        pass2 = request.POST['pass2']
        
        if User.objects.filter(username=username):
            messages.error(request, "Username already exist! Please try some other username.")
            return redirect('chatbot')
        
        if User.objects.filter(email=email).exists():
            messages.error(request, "Email Already Registered!!")
            return redirect('chatbot')
        
        if len(username)>20:
            messages.error(request, "Username must be under 20 charcters!!")
            return redirect('chatbot')
        
        if pass1 != pass2:
            messages.error(request, "Passwords didn't matched!!")
            return redirect('chatbot')
        
        if not username.isalnum():
            messages.error(request, "Username must be Alpha-Numeric!!")
            return redirect('chatbot')
        
        myuser = User.objects.create_user(username, email, pass1)
        myuser.first_name = fname
        myuser.last_name = lname
        myuser.is_active = False
        myuser.save()
        messages.success(request, "Your Account has been created succesfully!! Please check your email to confirm your email address in order to activate your account.")
        
        # Welcome Email
        subject = "Welcome to SGPT Login!!"
        message = "Hello " + myuser.first_name + "!! \n" + "Welcome to SGPT!! \nThank you for visiting our website\n. We have also sent you a confirmation email, please confirm your email address. \n\nThanking You"        
        from_email = settings.EMAIL_HOST_USER
        to_list = [myuser.email]
        send_mail(subject, message, from_email, to_list, fail_silently=True)
        
         # Email Address Confirmation Email
        current_site = get_current_site(request)
        email_subject = "Confirm your Email @ SGPT Login!!"
        message2 = render_to_string('email_confirmation.html',{
            
            'name': myuser.first_name,
            'domain': current_site.domain,
            'uid': urlsafe_base64_encode(force_bytes(myuser.pk)),
            'token': generate_token.make_token(myuser)
        })
        email = EmailMessage(
        email_subject,
        message2,
        settings.EMAIL_HOST_USER,
        [myuser.email],
        )
        email.fail_silently = True
        email.send()
        
        return redirect('signin')
    return render(request, 'register.html')



def logout(request):
    auth.logout(request)
    messages.success(request, "Logged Out Successfully!!")
    return redirect('signin')

def activate(request,uidb64,token):
    try:
        uid = force_text(urlsafe_base64_decode(uidb64))
        myuser = User.objects.get(pk=uid)
    except (TypeError,ValueError,OverflowError,User.DoesNotExist):
        myuser = None

    if myuser is not None and generate_token.check_token(myuser,token):
        myuser.is_active = True
        #user.profile.signup_confirmation = True
        myuser.save()
        login(request,myuser)
        messages.success(request, "Your Account has been activated!!")
        return redirect('signin')
    else:
        return render(request,'activation_failed.html')

def activation_failed(request):
    return render(request,'activation_failed.html')


def home(request):
    return render(request,'home.html')

def generate_assessment(message):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a Multiple Choice Questions generator bot. Output should contain question, options and asnwer in a JSON format. "},
            {"role": "user", "content": message},
        ]
    )
   

    answer = response.choices[0].message.content.strip()
    # Print the response for debugging
    print("Response from GPT-3:", answer)
    return answer

def assessment(request):
    if not request.user.is_authenticated:
        # Redirect the user to the login page, or return a suitable response
        return redirect('signin') 
    if request.method == 'POST':
        message = request.POST.get('message')
        response = generate_assessment(message)    
      
        # Print the response data for debugging
        print(response)

        # Parse the JSON response
        response_data = json.loads(response)

        # Clear existing AssessmentQuestion objects
        AssessmentQuestion.objects.all().delete()

        # Check if the response contains the "questions" key
        questions_data = response_data.get("questions", [])

        # Create AssessmentQuestion objects for each question
        for question_data in questions_data:
            question = question_data["question"]
            options = question_data.get("options", [])  # Use get() to handle missing keys gracefully
            answer = question_data.get("answer", "")  # Use get() to handle missing keys gracefully

            # Create and save the AssessmentQuestion object
            assessment_question = AssessmentQuestion.objects.create(
                question=question,
                options=options,
                answer=answer,
            )
            assessment_question.save()

        return JsonResponse({'message': message, 'response': response})
    return render(request, 'assessment.html')


def interface(request):
    # Fetch the first 5 questions for display
    assessment_questions = AssessmentQuestion.objects.all()[:10]
    
    if request.method == 'POST':
        user = request.user
        score = 0
        user_answers = []
        
        # Calculate the total number of questions for the max score
        max_score = len(assessment_questions)

        # Generate a unique assessment ID
        last_assessment_number = AssessmentHistory.objects.filter(user=request.user).count() + 1
        assessment_id = f"{user.username}{last_assessment_number}"
        
        for question in assessment_questions:
            selected_option_key = f'selected_options_{question.id}'
            submitted_answer = request.POST.get(selected_option_key, None)
            
            if submitted_answer == question.answer:
                score += 1
                answer_status = True
            else:
                answer_status = False
            
            # Collect user's answers and question details
            user_answers.append({
                'question': question.question,
                'correct_answer': question.answer,
                'user_answer': submitted_answer or "No answer",
                'answer_status': answer_status
            })
        
        # Convert the user answers to a JSON string
        result_details_json = json.dumps(user_answers)
        
        # Create a single AssessmentHistory instance for this assessment
        AssessmentHistory.objects.create(
            assessment_id=assessment_id,
            user=request.user,
            score=score,
            max_score=max_score,
            result_details=result_details_json
        )
        
        return render(request, 'score.html', {
            'score': score,
            'max_score': max_score,
            'user_answers': user_answers,
        })

    # Display the questions if the method is not POST
    return render(request, 'interface.html', {'assessment_questions': assessment_questions})

def one_line_interface(request):
    # Fetch the first 5 one-line questions for display
    one_line_questions = AssessmentQuestion.objects.all()[:5]  # Assuming OneLineQuestion is your model for one-line questions

    if request.method == 'POST':
        score = 0
        submitted_answers = []
        for question in one_line_questions:
            answer_key = f'answer_{question.id}'
            submitted_answer = request.POST.get(answer_key)
            submitted_answers.append(submitted_answer)
            correct_answer = question.answer
            if submitted_answer == correct_answer:
                score += 1
        
        # Render the score template with the score
        return render(request, 'score.html', {'score': score})

    return render(request, 'one_line_interface.html', {'one_line_questions': one_line_questions})


def true_n_false_interface(request):
    # Fetch the True or False questions for display
    true_false_questions = AssessmentQuestion.objects.all()[:10]  # Assuming TrueFalseQuestion is your model for True or False questions

    if request.method == 'POST':
        score = 0
        max_score = len(true_false_questions)
        incorrect_answers = {}
        submitted_answers = []
        user_answers = []
        all_answered = True
        
        # Generate a unique assessment ID
        last_assessment_number = AssessmentHistory.objects.filter(user=request.user).count() + 1
        assessment_id = f"assessment-{last_assessment_number}"
        
        for question in true_false_questions:
            answer_key = f'answer_{question.id}'
            submitted_answer = request.POST.get(answer_key)
            submitted_answers.append(submitted_answer)
            correct_answer = question.answer.lower()  # Convert to lowercase for case-insensitive comparison
            submitted_answer = submitted_answer.lower()# Convert to lowercase for case-insensitive comparison
            if submitted_answer == correct_answer:
                score += 1
            else:
                # Store incorrect answers along with correct options
                incorrect_answers[question.question] = {
                    'submitted_answer': submitted_answer,
                    'correct_answer': correct_answer,
                }
            question_data = {
                'question': question.question,
                'correct_answer': correct_answer,
                'user_answer': submitted_answer if submitted_answer else "No answer"
            }
            user_answers.append(question_data)
            
            if f'answer_{question.id}' not in request.POST or not request.POST.get(f'answer_{question.id}').strip():
                all_answered = False
                break
        
        if not all_answered:
            messages.error(request, 'Please select one option for each question.')
            return render(request, 'true_n_false_interface.html', {
                'true_false_questions': true_false_questions
            })

        # Convert the user answers to a JSON string
        result_details_json = json.dumps(user_answers)
        
        # Create a single AssessmentHistory instance for this assessment
        assessment_history = AssessmentHistory.objects.create(
            assessment_id=assessment_id,
            user=request.user,
            score=score,
            max_score=max_score,
            result_details=result_details_json
        )

        # Render the score template with the score
        return render(request, 'score.html', {'score': score, 'max_score': max_score, 'incorrect_answers': incorrect_answers})

    return render(request, 'true_n_false_interface.html', {'true_false_questions': true_false_questions})


def assessment_history(request):
    if not request.user.is_authenticated:
        # Redirect the user to the login page, or return a suitable response
        return redirect('signin') 
    # Fetch all assessment history records for the current user in reverse order from the AssessmentHistory database
    user_assessment_history = AssessmentHistory.objects.filter(user=request.user).order_by('-assessment_id')
    total_assessments = user_assessment_history.count()
    average_score = user_assessment_history.aggregate(Avg('score'))['score__avg'] or 0

    # Example data for graph (modify as needed)
    scores = list(user_assessment_history.values_list('score', flat=True))
     
    for history in user_assessment_history:
            # Parse the JSON string into a Python object
        history.result_details = json.loads(history.result_details)
        
    # Pass these to the context
    context = {
        'assessment_history': user_assessment_history,
        'total_assessments': total_assessments,
        'average_score': average_score,
        'scores': scores,
        
    }
    return render(request, 'Assessment_History.html', context)

#Uploading files for RAG
def upload_document(request):
    if not request.user.is_authenticated:
        # Redirect the user to the login page, or return a suitable response
        return redirect('signin') 
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            # Redirect to the same page to avoid form resubmission on refresh
            return redirect('upload_document')
    else:
        form = DocumentForm()
    
    # Fetch all uploaded documents
    documents = Document.objects.all()
    
    return render(request, 'upload_document.html', {'form': form, 'documents': documents})


from django.shortcuts import render, redirect
from django.http import JsonResponse
from .models import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

def rag_search(request):
    if not request.user.is_authenticated:
        # Redirect the user to the login page, or return a suitable response
        return redirect('signin') 
    # Retrieve all documents uploaded by the user
    documents = Document.objects.all()

    if request.method == 'POST':
        # Get the user's query and selected document ID from the form
        query = request.POST.get('query', '')
        selected_doc_id = request.POST.get('document')

        # Retrieve the selected document object
        selected_document = Document.objects.get(id=selected_doc_id)

        if selected_document.file.name.endswith('.txt'):
            # Load text from text file
            loader = TextLoader(selected_document.file.path, encoding="utf-8")
            loaded_text = loader.load()
            document_chunks = loaded_text
        elif selected_document.file.name.endswith('.pdf'):
            # Load text from PDF file
            loader = PyPDFLoader(selected_document.file.path)
            loaded_pdf = loader.load()
            document_chunks = loaded_pdf

        # Create a text splitter instance
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)

        # Split the document into smaller chunks
        document_chunks = text_splitter.split_documents(document_chunks)

        # Create a FAISS vector database from the documents
        db = FAISS.from_documents(document_chunks, OpenAIEmbeddings())

        # Load the GPT-3.5-turbo model
        llm = ChatOpenAI(model="gpt-3.5-turbo")

        # Design a chat prompt template
        prompt = ChatPromptTemplate.from_template(
            """ 
            Answer the following question based only on the provided context.
            Think step by step before providing a detailed answer.
            I will tip you $1000 if the user finds the answer helpful.
            <context>
            {context}
            </context>
            Question: {input}"""
        )

        # Create a document chain for processing documents
        document_chain = create_stuff_documents_chain(llm, prompt)

        # Convert the FAISS vector database to a retriever
        retriever = db.as_retriever()

        # Create a retrieval chain with the retriever and document chain
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Invoke the retrieval chain with the user's query
        response = retrieval_chain.invoke({"input": query})

        # Modify the response to include only 'input' and 'answer'
        response = {'input': response['input'], 'answer': response['answer']}

        # Pass the documents, query, and response to the template
        context = {
            'documents': documents,
            'query': query,
            'response': response,
        }
        return render(request, 'rag_search.html', context)

    # Pass the documents to the template for GET requests
    context = {
        'documents': documents,
    }
    return render(request, 'rag_search.html', context)

def rag_test(request):
    if not request.user.is_authenticated:
        # Redirect the user to the login page, or return a suitable response
        return redirect('signin') 
    # Retrieve all documents uploaded by the user
    documents = Document.objects.all()

    if request.method == 'POST':
        # Get the user's query and selected document ID from the form
        query = request.POST.get('query', '')
        selected_doc_id = request.POST.get('document')

        # Retrieve the selected document object
        selected_document = Document.objects.get(id=selected_doc_id)

        if selected_document.file.name.endswith('.txt'):
            # Load text from text file
            loader = TextLoader(selected_document.file.path, encoding="utf-8")
            loaded_text = loader.load()
            document_chunks = loaded_text
        elif selected_document.file.name.endswith('.pdf'):
            # Load text from PDF file
            loader = PyPDFLoader(selected_document.file.path)
            loaded_pdf = loader.load()
            document_chunks = loaded_pdf

        # Create a text splitter instance
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)

        # Split the document into smaller chunks
        document_chunks = text_splitter.split_documents(document_chunks)

        # Create a FAISS vector database from the documents
        db = FAISS.from_documents(document_chunks, OpenAIEmbeddings())

        # Load the GPT-3.5-turbo model
        llm = ChatOpenAI(model="gpt-3.5-turbo")

        # Design a chat prompt template
        prompt = ChatPromptTemplate.from_template(
            """ 
            Answer the following question based only on the provided context.
            Think step by step before providing a detailed answer.
            I will tip you $1000 if the user finds the answer helpful.
            <context>
            {context}
            </context>
            Question: {input}"""
        )

        # Create a document chain for processing documents
        document_chain = create_stuff_documents_chain(llm, prompt)

        # Convert the FAISS vector database to a retriever
        retriever = db.as_retriever()

        # Create a retrieval chain with the retriever and document chain
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Invoke the retrieval chain with the user's query
        response = retrieval_chain.invoke({"input": query})

        # Modify the response to include only 'input' and 'answer'
        response = {'input': response['input'], 'answer': response['answer']}

        # Pass the documents, query, and response to the template
        context = {
            'documents': documents,
            'query': query,
            'response': response,
        }
        return render(request, 'rag_test.html', context)

    # Pass the documents to the template for GET requests
    context = {
        'documents': documents,
    }
    return render(request, 'rag_test.html', context)

def dashboard(request):
    if not request.user.is_authenticated:
        # Redirect the user to the login page if not authenticated
        return redirect('signin')
    
    # Fetch the top 5 rows from the assessment_history table
    #recent_assessments = AssessmentHistory.objects.all().order_by('-date_taken')[:5]

    # Fetch the top 5 history records for the current user in reverse order from the AssessmentHistory database
    user_assessment_history = AssessmentHistory.objects.filter(user=request.user).order_by('-date_taken')[:5]
    total_assessments = user_assessment_history.count()
    average_score = user_assessment_history.aggregate(Avg('score'))['score__avg'] or 0

    # Example data for graph (modify as needed)
    scores = list(user_assessment_history.values_list('score', flat=True))
     
    for history in user_assessment_history:
            # Parse the JSON string into a Python object
        history.result_details = json.loads(history.result_details)
        
    # Pass these to the context
    context = {
        'assessment_history': user_assessment_history,
        'total_assessments': total_assessments,
        'average_score': average_score,
        'scores': scores,
        
    }
    
    return render(request, 'dashboard.html', context)