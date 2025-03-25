from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('AliceNBob/', views.AliceNBob, name='AliceNBob'),
    path('AliceBob/', views.AliceBob, name='AliceBob'),
    path('AliceMnist/', views.Alice_mnist, name='AliceMnist'),
    path('mis/', views.mis, name='mis'),
]

