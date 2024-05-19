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
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from io import BytesIO
import urllib
import base64
import seaborn as sns
import matplotlib
matplotlib.use('Agg')


# Pages
def index(request):
    years = [2021, 2022, 2023]
    average_values = {'toleransi': [], 'kesetaraan': [], 'kerjasama': []}
    
    for year in years:
        data_year = Kub.objects.filter(tahun=year)
        
        if data_year.exists():
            data_year_df = pd.DataFrame(list(data_year.values_list('provinsi', 'toleransi', 'kesetaraan', 'kerjasama')), columns=['provinsi', 'toleransi', 'kesetaraan', 'kerjasama'])
            grouped_data = data_year_df.groupby('provinsi').agg({
                'toleransi': 'mean',
                'kesetaraan': 'mean',
                'kerjasama': 'mean'
            }).reset_index()
            
            average_values['toleransi'].append(grouped_data['toleransi'].mean())
            average_values['kesetaraan'].append(grouped_data['kesetaraan'].mean())
            average_values['kerjasama'].append(grouped_data['kerjasama'].mean())

    return render(request, 'pages/index.html', { 
        'segment': 'index',
        'average_values': average_values
    })

class Cluster(View):
    def get(self, request, *args, **kwargs):
        bic_scores_df = self.calculate_bic_scores()

        years = bic_scores_df.columns
        bic_scores = bic_scores_df.values.T

        plt.figure(figsize=(5, 4))
        for i, year in enumerate(years):
            plt.plot(range(1, 10), bic_scores[i], label=year)

        plt.title('BIC Scores for Different Numbers of Clusters')
        plt.xlabel('Number of Clusters')
        plt.ylabel('BIC Score')
        plt.xticks(range(1, 10))
        plt.legend(title='Year')
        plt.grid(True)
        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()

        graphic = urllib.parse.quote(base64.b64encode(image_png))

        best_bic_clusters_html = f'<img src="data:image/png;base64,{graphic}" alt="BIC Scores Plot">'
        bic_scores_html = bic_scores_df.to_html()

        gmm_data = self.generate_gmm_plots(years)

        return render(request, 'pages/cluster.html', {
            'segment': 'cluster',
            'best_bic_clusters_html': best_bic_clusters_html,
            'bic_scores_html': bic_scores_html,
            'gmm_plots': gmm_data['plots'],
            'gmm_tables': gmm_data['tables'],
        })

    def calculate_bic_scores(self):
        bic_scores_by_year = {}
        start_year = 2021
        end_year = 2023
        years_range = range(start_year, end_year + 1)
        min_samples_per_cluster = 2

        for year in years_range:
            data_year = Kub.objects.filter(tahun=year)

            if data_year.count() >= min_samples_per_cluster:
                features = ['toleransi', 'kesetaraan', 'kerjasama']
                X = [[getattr(item, feature) for feature in features] for item in data_year]
                data_year_df = pd.DataFrame(X, columns=features)
                imputer = SimpleImputer(strategy='mean')
                X = imputer.fit_transform(data_year_df)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                bic_scores = []

                for n_components in range(1, 10):
                    if data_year.count() < min_samples_per_cluster * n_components:
                        continue

                    gmm = GaussianMixture(n_components=n_components, random_state=42)
                    gmm.fit(X_scaled)
                    bic = gmm.bic(X_scaled)
                    bic_scores.append(bic)

                bic_scores_by_year[year] = bic_scores

        bic_scores_df = pd.DataFrame(bic_scores_by_year)
        return bic_scores_df

    def generate_gmm_plots(self, years):
        plots = {}
        tables = {}
        best_clusters = {2021: 8, 2022: 6, 2023: 9}

        for year in years:
            data_year = Kub.objects.filter(tahun=year)

            if data_year.exists():
                data_year_df = pd.DataFrame(list(data_year.values_list('id', 'provinsi', 'toleransi', 'kesetaraan', 'kerjasama')), columns=['id', 'provinsi', 'toleransi', 'kesetaraan', 'kerjasama'])
                grouped_data = data_year_df.groupby('provinsi').agg({
                    'toleransi': 'mean',
                    'kesetaraan': 'mean',
                    'kerjasama': 'mean'
                }).reset_index()
                imputer = SimpleImputer(strategy='mean')
                X = imputer.fit_transform(grouped_data[['toleransi', 'kesetaraan', 'kerjasama']])
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                n_clusters = best_clusters[year]
                gmm = GaussianMixture(n_components=n_clusters, random_state=42)
                gmm.fit(X_scaled)
                labels = gmm.predict(X_scaled)
                grouped_data['cluster'] = labels
                grouped_data = self.reassign_clusters(grouped_data, n_clusters)

                plt.figure(figsize=(14, 6))
                sns.scatterplot(x='toleransi', y='kesetaraan', hue='cluster', data=grouped_data, palette='Set1', style='cluster', markers=True)
                plt.title(f'Clusters for year {year}')
                plt.xlabel('Toleransi')
                plt.ylabel('Kesetaraan')
                plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
                
                buffer = BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                image_png = buffer.getvalue()
                buffer.close()

                graphic = base64.b64encode(image_png).decode('utf-8')
                plots[year] = f'data:image/png;base64,{graphic}'
                plt.close()

                tables[year] = grouped_data.to_dict(orient='records')

        return {'plots': plots, 'tables': tables}

    def reassign_clusters(self, data, n_clusters):
        cluster_means = data.groupby('cluster')['toleransi'].mean().sort_values().reset_index()
        cluster_means['new_cluster'] = range(1, n_clusters + 1)
        cluster_mapping = cluster_means.set_index('cluster')['new_cluster'].to_dict()
        data['cluster'] = data['cluster'].map(cluster_mapping)
        return data

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
            "segment": "preprocessing",
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