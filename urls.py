from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.home_view, name='home'),
    path('predict/', views.predict_view, name='predict'),
    path('history/', views.history_view, name='history'),
    path('download_csv/', views.download_csv, name='download_csv'),
    path('download_pdf/', views.download_pdf, name='download_pdf'),
    path('camera/', views.camera_predict, name='camera_predict'),
    path('visualization/', views.visualization_view, name='visualization'),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
