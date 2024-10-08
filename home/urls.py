from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('', views.index, name='index'),
    path('cluster/', views.Cluster.as_view(), name='cluster'),
    path('preprocessing/', views.Preprocessing.as_view(), name='preprocessing'),
    path('visualize/', views.Visualize.as_view(), name='visualize'),
    
    path('export/csv/', views.ExportCSV.as_view(), name='export_csv'),
    
    path('api/cluster-geojson/', views.cluster_geojson, name='cluster_geojson'),
    path('data-agama/', views.data_agama, name='data_agama'),
    path('data-jenis-kelamin/', views.data_jenis_kelamin, name='data_jenis_kelamin'),
    path('data-usia/', views.data_usia, name='data_usia'),
   
    path('profile/', views.profile, name='profile'),

    # Authentication
    path('accounts/login/', views.UserLoginView.as_view(), name='login'),
    path('accounts/logout/', views.logout_view, name='logout'),
    path('accounts/register/', views.register, name='register'),
    path('accounts/password-change/', views.UserPasswordChangeView.as_view(), name='password_change'),
    path('accounts/password-change-done/', auth_views.PasswordChangeDoneView.as_view(
        template_name='accounts/password_change_done.html'
    ), name="password_change_done"),
    path('accounts/password-reset/', views.UserPasswordResetView.as_view(), name='password_reset'),
    path('accounts/password-reset-confirm/<uidb64>/<token>/', 
        views.UserPasswordResetConfirmView.as_view(), name='password_reset_confirm'),
    path('accounts/password-reset-done/', auth_views.PasswordResetDoneView.as_view(
        template_name='accounts/password_reset_done.html'
    ), name='password_reset_done'),
    path('accounts/password-reset-complete/', auth_views.PasswordResetCompleteView.as_view(
        template_name='accounts/password_reset_complete.html'
    ), name='password_reset_complete'),
]
