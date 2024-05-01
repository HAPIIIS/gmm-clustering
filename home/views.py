from django.shortcuts import render, redirect
from django.contrib.auth.views import LoginView, PasswordResetView, PasswordChangeView, PasswordResetConfirmView
from .forms import RegistrationForm, LoginForm, UserPasswordResetForm, UserSetPasswordForm, UserPasswordChangeForm
from django.contrib.auth import logout
from csv import reader
from io import TextIOWrapper
from .forms import UploadForm, KubForm
from django.views.generic.base import View
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from .models import Kub
import pandas as pd
from django.db.models import Q


# Pages
def index(request):
  return render(request, 'pages/index.html', { 'segment': 'index' })

def billing(request):
  return render(request, 'pages/billing.html', { 'segment': 'billing' })

class Preprocessing(View):
    def get(self, request, *args, **kwargs):
        # Get the search query from the GET parameters
        search_query = request.GET.get('search', '')

        # Fetch and filter records based on the search query
        records = Kub.objects.filter(
            Q(provinsi__icontains=search_query) |
            Q(kabupaten__icontains=search_query) |
            Q(agama__icontains=search_query) |
            Q(tahun__icontains=search_query)
            # Add more fields to search as needed
        ).order_by('id')

        # Paginate the filtered records
        paginator = Paginator(records, 10)  # 10 records per page
        page_number = request.GET.get('page')
        try:
            paginated_records = paginator.page(page_number)
        except PageNotAnInteger:
            paginated_records = paginator.page(1)
        except EmptyPage:
            paginated_records = paginator.page(paginator.num_pages)

        return render(request, 'pages/preprocessing.html', {
            "form": UploadForm(),
            "paginated_records": paginated_records,
            "search_query": search_query,
            "page": paginated_records.number  # Add the page number to the context
        })

    def post(self, request, *args, **kwargs):
        kub_file = request.FILES.get("kub_file")
        if kub_file:
            # Save the uploaded CSV file temporarily
            filename = kub_file.name
            with open(filename, 'wb+') as destination:
                for chunk in kub_file.chunks():
                    destination.write(chunk)

            # Process the CSV file and upload data to the database
            with open(filename, 'r') as cleaned_csv:
                csv_reader = reader(cleaned_csv)
                next(csv_reader)  # Skip the header row

                for row in csv_reader:
                    row_data = row[0].split(',')
                    if len(row_data) != 13:
                        print(f"Row has incorrect number of values: {row_data}")
                        continue

                    provinsi, kabupaten, jenis_kelamin, usia, agama, menikah, pendidikan, pekerjaan, kub, toleransi, kesetaraan, kerjasama, tahun = row_data

                    # Handle 'NA' values for numeric columns
                    if jenis_kelamin == 'NA':
                        jenis_kelamin = 0
                    else:
                        jenis_kelamin = int(jenis_kelamin)

                    if usia == 'NA':
                        usia = 0
                    else:
                        usia = int(usia)

                    if menikah == 'NA':
                        menikah = 0
                    else:
                        menikah = int(menikah)

                    if pendidikan == 'NA':
                        pendidikan = 0
                    else:
                        pendidikan = int(pendidikan)

                    if pekerjaan == 'NA':
                        pekerjaan = 0
                    else:
                        pekerjaan = int(pekerjaan)

                    # Handle 'NA' values for float columns
                    if kub == 'NA':
                        kub = 0.0
                    else:
                        kub = float(kub)

                    if toleransi == 'NA':
                        toleransi = 0.0
                    else:
                        toleransi = float(toleransi)

                    if kesetaraan == 'NA':
                        kesetaraan = 0.0
                    else:
                        kesetaraan = float(kesetaraan)

                    if kerjasama == 'NA':
                        kerjasama = 0.0
                    else:
                        kerjasama = float(kerjasama)

                    tahun = int(tahun)

                    # Convert data types
                    jenis_kelamin = int(jenis_kelamin)
                    usia = int(usia)
                    menikah = int(menikah)
                    pendidikan = int(pendidikan)
                    pekerjaan = int(pekerjaan)
                    kub = float(kub)
                    toleransi = float(toleransi)
                    kesetaraan = float(kesetaraan)
                    kerjasama = float(kerjasama)
                    tahun = int(tahun)

                    # Apply data filtering and cleaning
                    if (jenis_kelamin not in [1, 2]) or (menikah not in range(1, 5)) or (tahun not in range(2021, 2024)):
                        continue

                    if pd.isna(provinsi):
                        provinsi = cleaned_csv.provinsi.mean()

                    if pd.isna(kabupaten):
                        kabupaten = cleaned_csv.kabupaten.mean()

                    if pd.isna(pendidikan):
                        pendidikan = (pendidikan % 13) if (pendidikan % 13 != 0) else 12

                    form_data = {
                        "provinsi": provinsi,
                        "kabupaten": kabupaten,
                        "jenis_kelamin": jenis_kelamin,
                        "usia": usia,
                        "agama": agama,
                        "menikah": menikah,
                        "pendidikan": pendidikan,
                        "pekerjaan": pekerjaan,
                        "kub": kub,
                        "toleransi": toleransi,
                        "kesetaraan": kesetaraan,
                        "kerjasama": kerjasama,
                        "tahun": tahun
                    }
                    form = KubForm(data=form_data)
                    if form.is_valid():
                        form.save()
                    else:
                        return render(request, 'pages/preprocessing.html', {
                            "form": UploadForm(),
                            "form_errors": form.errors,
                        })

            # Redirect to a success page or render a success message
            return render(request, 'pages/preprocessing.html', {"form": UploadForm(), "success_message": "Data uploaded successfully."}, { 'segment': 'preprocessing' })
        else:
            return render(request, 'pages/preprocessing.html', {"form": UploadForm(), "file_missing": True}, { 'segment': 'preprocessing' })

def profile(request):
  if request.user.is_authenticated:
    username = request.user.username
  else:
    username = None

  return render(request, 'pages/profile.html', { 'segment': 'profile', 'username': username })


# Authentication
class UserLoginView(LoginView):
  template_name = 'accounts/login.html'
  form_class = LoginForm

def register(request):
  if request.method == 'POST':
    form = RegistrationForm(request.POST)
    if form.is_valid():
      form.save()
      print('Account created successfully!')
      return redirect('/accounts/login/')
    else:
      print("Register failed!")
  else:
    form = RegistrationForm()

  context = { 'form': form }
  return render(request, 'accounts/register.html', context)

def logout_view(request):
  logout(request)
  return redirect('/accounts/login/')

class UserPasswordResetView(PasswordResetView):
  template_name = 'accounts/password_reset.html'
  form_class = UserPasswordResetForm

class UserPasswordResetConfirmView(PasswordResetConfirmView):
  template_name = 'accounts/password_reset_confirm.html'
  form_class = UserSetPasswordForm

class UserPasswordChangeView(PasswordChangeView):
  template_name = 'accounts/password_change.html'
  form_class = UserPasswordChangeForm