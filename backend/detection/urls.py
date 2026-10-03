from django.urls import path
from . import views


urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("prediction/", views.prediction, name="prediction"),
    path("batch-analysis/", views.batch_analysis, name="batch_analysis"),
    path("model-information/", views.model_information, name="model_information"),
]