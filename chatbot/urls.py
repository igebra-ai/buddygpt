from django.urls import path
from . import views
from django.contrib.auth import views as auth_views


urlpatterns = [
    #path('', views.home, name='home'),
    path('assessment', views.assessment, name='assessment'),
    path('signin', views.signin, name='signin'),
    path('chat', views.chat, name='chat'),
    path('chatbot', views.chatbot, name='chatbot'),
    
    path('edit-profile/', views.edit_profile, name='edit-profile'),
    path('profile/', views.view_profile, name='view-profile'),

    
    #path('signin', views.signin, name='signin'),
    path('register', views.register, name='register'),
    path('logout', views.logout, name='logout'),
    path('activation_failed', views.activation_failed, name='activation_failed'),
    path('activate/<uidb64>/<token>', views.activate, name='activate'),
    path('', views.about, name='about'),

    path('interface/', views.interface, name='interface'),
    path('interface/assessment_history', views.assessment_history, name='assessment_history'),
    path('one_line_interface/', views.one_line_interface, name='one_line_interface'),
    path('true_n_false_interface/', views.true_n_false_interface, name='true_n_false_interface'),
    path('assessment_history/', views.assessment_history, name='assessment_history'),
    path('assessment_history/assessment', views.assessment, name='assessment'),
    #path('assessment_history/assessment_history', views.assessment_history, name='assessment_history'),
    path('interface/assessment', views.assessment, name='assessment'),
    path('assessment', views.assessment, name='assessment'),
    path('dashboard/', views.dashboard, name='dashboard'),
    #path('dashboard/assessment', views.assessment, name='assessment'),

    path('upload_document/', views.upload_document, name='upload_document'),
    path('rag_search', views.rag_search, name='rag_search'),
    path('rag_test', views.rag_test, name='rag_test'),
    #path('upload_document/dashboard', views.dashboard, name='dashboard'),
    #path('upload_document/assessment', views.assessment, name='assessment'),
    #path('upload_document/assessment_history', views.assessment_history, name='assessment_history'),


    path('password_reset/', auth_views.PasswordResetView.as_view(template_name='password_reset_form.html'), name='password_reset'),
    path('password_reset_done/', auth_views.PasswordResetDoneView.as_view(template_name='password_reset_done.html'), name='password_reset_done'),
    path('password_reset_confirm/<uidb64>/<token>/', auth_views.PasswordResetConfirmView.as_view(template_name='password_reset_confirm.html'), name='password_reset_confirm'),
    path('password_reset_complete', auth_views.PasswordResetCompleteView.as_view(template_name='password_reset_complete.html'), name='password_reset_complete'),
]
