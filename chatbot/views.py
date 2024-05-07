from django.shortcuts import render, redirect
from .forms import DocumentForm
from django.http import JsonResponse
from openai import OpenAI
from django.contrib import auth
from django.contrib.auth.models import User
from .models import AssessmentQuestion, AssessmentHistory, Document, AssessmentSubject, AssessmentTopic, AssessmentFormat,Question,Answer
from django.shortcuts import render
from django.http import JsonResponse
import os
from django.db.models import Avg,Sum,Max
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
            {"role": "system", "content": "You are a Multiple Choice Questions generator bot. Output should contain question, options and answer in a JSON format. The value for key 'options' should be a python list."},
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
        subject = request.POST.get('subject')
        topic = request.POST.get('topic')
        assess_type = request.POST.get('type')

        response = generate_assessment(message)

        # Print the response data for debugging
        print(response)

        # Parse the JSON response
        response_data = json.loads(response)

        # Clear existing AssessmentQuestion objects
        AssessmentQuestion.objects.all().delete()

        # Clear existing AssessmentSubject objects
        AssessmentSubject.objects.all().delete()

        # Clear existing AssessmentTopic objects
        AssessmentTopic.objects.all().delete()

        # Create a new AssessmentSubject instance
        assessment_subject = AssessmentSubject.objects.create(subject=subject)

        # Create a new AssessmentSubject instance
        assessment_topic = AssessmentTopic.objects.create(topic=topic)

        # Create a new AssessmentSubject instance
        assessment_format = AssessmentFormat.objects.create(format=assess_type)

        # Print the created subject,topic,format for debugging
        print("Subject:", assessment_subject.subject)
        print("Topic:", assessment_topic.topic)
        print("Format:", assessment_format.format)

        # Check if the response contains the "questions" key
        questions_data = response_data.get("questions", [])############### Let's add subject, topics, format
                                                                                    ######### in AssessmentQuestion models

        # Create AssessmentQuestion objects for each question
        for question_data in questions_data:
            Question = question_data["question"]
            Options = question_data.get("options", [])  # Use get() to handle missing keys gracefully
            Answer = question_data.get("answer", "")  # Use get() to handle missing keys gracefully

            # Create and save the AssessmentQuestion object
            assessment_question = AssessmentQuestion.objects.create(
                question=Question,
                options=Options,
                answer=Answer,
            )
            assessment_question.save()

        return JsonResponse({'message': message, 'response': response, 'subject': subject, 'topic': topic, 'format': assess_type})

    return render(request, 'assessment.html')


def interface(request):

    assessment_questions = AssessmentQuestion.objects.all()[:10]
    assessment_subject = AssessmentSubject.objects.first()  # Assuming you want to fetch the first subject
    assessment_topic = AssessmentTopic.objects.first()
    assessment_format = AssessmentFormat.objects.last()

    if request.method == 'POST':
        user = request.user
        score = 0
        user_answers = []
        #topic = request.POST.get('topic')
        #assess_type = request.POST.get('assess_type')

        # Calculate the total number of questions for the max score
        max_score = len(assessment_questions)

        # Generate a unique assessment ID
        last_assessment_number = AssessmentHistory.objects.filter(user=request.user).count() + 1
        assessment_id = f"{user.username}-{last_assessment_number}"

        subject = assessment_subject.subject if assessment_subject else None
        topic = assessment_topic.topic if assessment_topic else None
        format = assessment_format.format if assessment_topic else None

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
            result_details=result_details_json,
            subject=subject,
            topic=topic,
            type=format
        )

        print("Subject:", subject)
        print("Topic:", topic)
        print("Format:", format)

        return render(request, 'score.html', {
            'score': score,
            'max_score': max_score,
            'user_answers': user_answers,
        })
        
    
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
    assessment_subject = AssessmentSubject.objects.first()  # Assuming you want to fetch the first subject
    assessment_topic = AssessmentTopic.objects.first()
    assessment_format = AssessmentFormat.objects.last()
    
    if request.method == 'POST':
        user = request.user
        score = 0
        max_score = len(true_false_questions)
        incorrect_answers = {}
        submitted_answers = []
        user_answers = []
        all_answered = True

        # Generate a unique assessment ID
        last_assessment_number = AssessmentHistory.objects.filter(user=request.user).count() + 1
        assessment_id = f"{user.username}-{last_assessment_number}"
        
        subject = assessment_subject.subject if assessment_subject else None
        topic = assessment_topic.topic if assessment_topic else None
        format = assessment_format.format if assessment_topic else None

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
            result_details=result_details_json,
            subject=subject,
            topic=topic,
            type=format
        )
        
        print("Subject:", subject)
        print("Topic:", topic)
        print("Format:", format)

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
    
    # Calculate total obtained score and total maximum possible score
    total_obtained_score = user_assessment_history.aggregate(Sum('score'))['score__sum'] or 0
    total_max_score = user_assessment_history.aggregate(Sum('max_score'))['max_score__sum'] or 0
    
    # Calculate overall score percentage
    overall_score_percentage = (total_obtained_score / total_max_score) * 100 if total_max_score > 0 else 0
    
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
        'overall_score_percentage': overall_score_percentage,
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
from langchain_community.document_loaders.csv_loader import CSVLoader 
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
        elif selected_document.file.name.endswith('.csv'):
            # Load text from PDF file
            loader = CSVLoader(selected_document.file.path, encoding="utf-8")
            loaded_csv = loader.load()
            document_chunks = loaded_csv

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
        query = request.POST.get('combinedMessage', '')
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
        elif selected_document.file.name.endswith('.csv'):
            # Load text from PDF file
            loader = CSVLoader(selected_document.file.path, encoding="utf-8")
            loaded_csv = loader.load()
            document_chunks = loaded_csv

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
            You are a Questions generator bot according to context provided by user. Output should contain question, options and answer in a JSON format. The value for key 'options' should be a python list.
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
        response = { 'answer': response['answer']}
        
        print(response)
        
        # Clear existing AssessmentQuestion objects
        AssessmentQuestion.objects.all().delete()

        
        # Assuming the response from the LLM is a JSON string that needs to be parsed
        # Let's assume response['answer'] is already a dictionary for simplicity
        if isinstance(response['answer'], str):
            answer_data = json.loads(response['answer'])
        else:
            answer_data = response['answer']

        # Check if the response contains the "questions" key
        questions_data = answer_data.get("questions", [])############### Let's add subject, topics, format
                                                                                    ######### in AssessmentQuestion models

        # Create AssessmentQuestion objects for each question
        for question_data in questions_data:
            Question = question_data["question"]
            Options = question_data.get("options", [])  # Use get() to handle missing keys gracefully
            Answer = question_data.get("answer", "")  # Use get() to handle missing keys gracefully

            # Create and save the AssessmentQuestion object
            assessment_question = AssessmentQuestion.objects.create(
                question=Question,
                options=Options,
                answer=Answer,
            )
            assessment_question.save()
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

def rag_MCQInterface(request):
    
    assessment_questions = AssessmentQuestion.objects.all()[:10]
    

    if request.method == 'POST':
        user = request.user
        score = 0
        user_answers = []
        #topic = request.POST.get('topic')
        #assess_type = request.POST.get('assess_type')

        # Calculate the total number of questions for the max score
        max_score = len(assessment_questions)

        # Generate a unique assessment ID
        last_assessment_number = AssessmentHistory.objects.filter(user=request.user).count() + 1
        assessment_id = f"{user.username}-{last_assessment_number}"

        
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

       

        return render(request, 'score.html', {
            'score': score,
            'max_score': max_score,
            'user_answers': user_answers,
        })
        
    
    return render(request, 'rag_mcq.html', {'assessment_questions': assessment_questions})

def rag_TFinterface(request):
    # Fetch the True or False questions for display
    true_false_questions = AssessmentQuestion.objects.all()[:10]  # Assuming TrueFalseQuestion is your model for True or False questions
    
    
    if request.method == 'POST':
        user = request.user
        score = 0
        max_score = len(true_false_questions)
        incorrect_answers = {}
        submitted_answers = []
        user_answers = []
        all_answered = True

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


        # Render the score template with the score
        return render(request, 'score.html', {'score': score, 'max_score': max_score, 'incorrect_answers': incorrect_answers})

    return render(request, 'rag_TF.html', {'true_false_questions': true_false_questions})


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


from django.shortcuts import render, redirect
from .forms import ProfileForm
from .models import Profile
from django.contrib.auth.decorators import login_required

@login_required
def edit_profile(request):
    if request.method == 'POST':
        form = ProfileForm(request.POST, request.FILES, instance=request.user.profile)
        if form.is_valid():
            form.save()
            return redirect('dashboard')  # Redirect to a relevant page after saving
    else:
        form = ProfileForm(instance=request.user.profile)

    return render(request, 'edit_profile.html', {'form': form})

@login_required
def view_profile(request):
    profile = Profile.objects.get(user=request.user)  # Fetch the profile for the logged-in user
    return render(request, 'profile.html', {'profile': profile})

def about(request):
    return render(request, 'about.html')

def report(request):
    if not request.user.is_authenticated:
        return redirect('signin')

    # Calculating both average and total scores for each subject or type
    subjects_scores = AssessmentHistory.objects.filter(user=request.user) \
                                               .values('subject') \
                                               .annotate(average_score=Sum('score'),
                                                         total_score=Sum('max_score')) \
                                               .order_by('subject')

    subjects = [score['subject'] for score in subjects_scores]
    avg_scores = [score['average_score'] for score in subjects_scores]
    total_scores = [score['total_score'] for score in subjects_scores]
    
    # Add percentage calculation
    for score in subjects_scores:
        score['percentage'] = (score['average_score'] / score['total_score'] * 100) if score['total_score'] > 0 else 0
    
    type_scores = AssessmentHistory.objects.filter(user=request.user) \
                                           .values('type') \
                                           .annotate(average_score=Sum('score'),
                                                     total_score=Sum('max_score')) \
                                           .order_by('type')
                                           
    types = [score['type'] for score in type_scores]
    avg_scoress = [score['average_score'] for score in type_scores]
    total_scoress= [score['total_score'] for score in type_scores]
    
    # Add percentage calculation
    for score in type_scores:
        score['percentage'] = (score['average_score'] / score['total_score'] * 100) if score['total_score'] > 0 else 0

    return render(request, 'report.html', {
        'subjects': subjects,
        'avg_scores': avg_scores,
        'total_scores': total_scores,
        'types': types,
        'avg_scoress': avg_scoress,
        'total_scoress': total_scoress,
        'subjects_scores': subjects_scores,
        'types_scores': type_scores,
    })

def dash_report(request):
    if not request.user.is_authenticated:
        return redirect('signin')

    # Calculating both average and total scores for each subject or type
    subjects_scores = AssessmentHistory.objects.filter(user=request.user) \
                                               .values('subject') \
                                               .annotate(average_score=Sum('score'),
                                                         total_score=Sum('max_score')) \
                                               .order_by('subject')

    subjects = [score['subject'] for score in subjects_scores]
    avg_scores = [score['average_score'] for score in subjects_scores]
    total_scores = [score['total_score'] for score in subjects_scores]
    
    # Add percentage calculation
    for score in subjects_scores:
        score['percentage'] = (score['average_score'] / score['total_score'] * 100) if score['total_score'] > 0 else 0
    
    type_scores = AssessmentHistory.objects.filter(user=request.user) \
                                           .values('type') \
                                           .annotate(average_score=Sum('score'),
                                                     total_score=Sum('max_score')) \
                                           .order_by('type')
                                           
    types = [score['type'] for score in type_scores]
    avg_scoress = [score['average_score'] for score in type_scores]
    total_scoress= [score['total_score'] for score in type_scores]
    
    # Add percentage calculation
    for score in type_scores:
        score['percentage'] = (score['average_score'] / score['total_score'] * 100) if score['total_score'] > 0 else 0

    return render(request, 'report copy.html', {
        'subjects': subjects,
        'avg_scores': avg_scores,
        'total_scores': total_scores,
        'types': types,
        'avg_scoress': avg_scoress,
        'total_scoress': total_scoress,
        'subjects_scores': subjects_scores,
        'types_scores': type_scores,
    })

def openai_recommendation(message):
    system_message = """You are a test performance analyst.
You'll have to analyze the test history data for each subject and Based on the user's wrong responses to question, you can judge which concept does the user lacks and generate more such questions to the user for their practice.
You'll get the details of a single test followed by the questions and responses of the user and correct answer following it. Use proper spacing and line vreaks in your response.""" 
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": message},
        ]
    )

    answer = response.choices[0].message.content.strip()
    return answer

def recommend(request):
    if not request.user.is_authenticated:
        # Redirect the user to the login page, or return a suitable response
        return redirect('signin')
    if request.method == 'POST':
        message = request.POST.get('message')
        response = openai_recommendation(message)

        return JsonResponse({'message': message, 'response': response})
    return render(request, 'recommendation.html')

def dashboard_recommend(request, question_id=None):
    if not question_id:
        # Start with the first question
        question = Question.objects.first()
    else:
        question = Question.objects.get(id=question_id)

    if request.method == 'POST':
        answer_id = request.POST.get('answer')
        answer = Answer.objects.get(id=answer_id)
        if answer.redirect_url:
            return redirect(answer.redirect_url)
        # Proceed to the next question, if there is one related to this answer
        next_question = answer.next_question if hasattr(answer, 'next_question') else None
        if next_question:
            return redirect('question', question_id=next_question.id)
        else:
            return render(request, 'end.html', {'answer': answer})
        
    if not request.user.is_authenticated:
        return redirect('signin')

    # Calculate the average and total scores, and then the percentage
    subjects_scores = AssessmentHistory.objects.filter(user=request.user) \
                                               .values('subject') \
                                               .annotate(total_score=Sum('max_score'),
                                                         total_sum=Sum('score'))

    # Calculate percentage and add it to each item
    for item in subjects_scores:
        if item['total_score'] > 0:
            item['percentage'] = (item['total_sum'] / item['total_score']) * 100
        else:
            item['percentage'] = 0

    # Find the lowest scoring subject
    if subjects_scores:
        lowest_scoring_subject = min(subjects_scores, key=lambda x: x['percentage'])
    else:
        lowest_scoring_subject = None

    # Sort subjects by percentage in descending order for additional display
    sorted_subjects = sorted(subjects_scores, key=lambda x: x['percentage'], reverse=True)

    

    return render(request, 'recommendation copy.html', {'question': question,'sorted_subjects': sorted_subjects,
        'lowest_scoring_subject': lowest_scoring_subject})


def end(request):
    return render(request, 'end.html')