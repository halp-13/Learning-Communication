from django.urls import path

from . import views

urlpatterns = [

    path('', views.index, name='index'),
    path('AliceNBob/', views.AliceNBob, name='AliceNBob'),
    path('AliceBob/', views.AliceBob, name='AliceBob'),


]

