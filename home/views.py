from django.shortcuts import render, redirect
from django.contrib.auth.views import LoginView, PasswordResetView, PasswordChangeView, PasswordResetConfirmView
from .forms import RegistrationForm, LoginForm, UserPasswordResetForm, UserSetPasswordForm, UserPasswordChangeForm, UploadForm, KubForm, UploadClusterForm
from django.contrib.auth import logout
from csv import reader
from io import TextIOWrapper
from django.views.generic.base import View
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from .models import Kub, ClusterModel, Province
import pandas as pd
from django.db.models import Count, Q
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
from django.core.serializers import serialize
from django.http import JsonResponse, HttpResponse
import csv

# Halaman Dashboard digunakan untuk menampilkan grafik Indeks KUB dalam range 3 tahun
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

# Cluster GeoJSON digunakan untuk fetch data dari tabel clustermodel ke GeoJSON untuk dicocokkan nama provinsinya lalu di append data clusternya
def cluster_geojson(request):
    clusters = ClusterModel.objects.all()
    cluster_data = []
   
    for cluster in clusters:
        data = {
            "province": cluster.province.name,
            "year": cluster.year,
            "cluster": cluster.cluster,
            "toleransi": cluster.toleransi,
            "kesetaraan": cluster.kesetaraan,
            "kerjasama": cluster.kerjasama
        }
        cluster_data.append(data)
   
    return JsonResponse({"clusters": cluster_data})

# Data agama untuk menampilkan statistik responden dari tabel KUB
def data_agama(request):
    agama_counts = Kub.objects.values('agama').annotate(count=Count('agama')).order_by('-count')
    total_count = Kub.objects.count()
    data = {item['agama']: item['count'] for item in agama_counts}
    response_data = {
        'data': data,
        'total_count': total_count
    }
    return JsonResponse(response_data)

# Data Jenis kelamin digunakan untuk menampilkan statistik responden dari tabel KUB di dashboard
def data_jenis_kelamin(request):
    jenis_kelamin_counts = Kub.objects.values('jenis_kelamin').annotate(count=Count('jenis_kelamin')).order_by('-count')
    data = {item['jenis_kelamin']: item['count'] for item in jenis_kelamin_counts}
    return JsonResponse(data)

# Data usia digunakan untuk menampilkan statistik responden dari tabel KUB di dashboard
def data_usia(request):
    usia_ranges = [
        (10, 19),
        (20, 29),
        (30, 39),
        (40, 49),
        (50, 59),
        (60, 69),
        (70, 79),
        (80, 89),
    ]

    usia_counts = {f"{start}-{end}": 0 for start, end in usia_ranges}
    for start, end in usia_ranges:
        count = Kub.objects.filter(usia__gte=start, usia__lte=end).count() # GTE stands for greater than equal (>=), LTE less than equal (<=)
        usia_counts[f"{start}-{end}"] = count

    return JsonResponse(usia_counts)

# Halaman clustering
class Cluster(View):
    def get(self, request, *args, **kwargs):
        bic_scores_df = self.calculate_bic_scores() # Mengambil hasil perhitungan dari function calculate_bic_scores dibawah

        selected_year = int(request.GET.get('year', 2021))
        selected_clusters = int(request.GET.get('n_clusters', 8))
        selected_years = [selected_year]

        plt.figure(figsize=(8, 4))
        for year in selected_years:
            if year in bic_scores_df.columns:
                bic_scores = bic_scores_df[year] # Menyimpan data bic score dalam dataframe
                plt.plot(bic_scores.index, bic_scores, marker='o', label=f'Year {year}')
                plt.scatter([selected_clusters], [bic_scores[selected_clusters]], color='red')
                plt.text(selected_clusters, bic_scores[selected_clusters], f'{bic_scores[selected_clusters]:.2f}', ha='center', va='bottom', color='red') # BIC scores yang dipilih akan diberikan penanda merah beserta scorenya

        plt.title('BIC Score for Selected Number of Clusters')
        plt.xlabel('Number of Clusters')
        plt.ylabel('BIC Score')
        plt.xticks(range(1, 10)) # Iterasi range jumlah cluster dari 1-9
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()

        graphic = base64.b64encode(image_png).decode('utf-8')
        best_bic_clusters_html = f'<img src="data:image/png;base64,{graphic}" alt="BIC Scores Plot">'
        bic_scores_html = bic_scores_df[selected_years].to_html()  # Hanya menampilkan data berdasarkan tahun yang dipilih

        gmm_data = self.generate_gmm_plots([selected_year], selected_clusters)

        return render(request, 'pages/cluster.html', {
            'segment': 'cluster',
            'years': [2021, 2022, 2023],
            'selected_year': selected_year,
            'cluster_range': range(1, 10),
            'selected_clusters': selected_clusters,
            'best_bic_clusters_html': best_bic_clusters_html,
            'bic_scores_html': bic_scores_html,
            'gmm_plot': gmm_data['plots'].get(selected_year),
            'gmm_table': gmm_data['tables'].get(selected_year),
        })

    # Perhitungan BIC Scores pertahun dimulai dari 2021-2023
    def calculate_bic_scores(self):
        bic_scores_by_year = {}
        start_year = 2021
        end_year = 2023
        years_range = range(start_year, end_year + 1) #Harus di +1 untuk memasukkan end year ke dalam iterasi
        min_samples_per_cluster = 2 # Minimal sample data untuk dilakukan perhitungan BIC

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

        bic_scores_df = pd.DataFrame(bic_scores_by_year, index=range(1, 10))
        return bic_scores_df

    def generate_gmm_plots(self, years, selected_clusters):
        plots = {}
        tables = {}

        for year in years:
            data_year = Kub.objects.filter(tahun=year)

            if data_year.exists():
                data_year_df = pd.DataFrame(list(data_year.values_list('id', 'provinsi', 'toleransi', 'kesetaraan', 'kerjasama')), columns=['id', 'provinsi', 'toleransi', 'kesetaraan', 'kerjasama'])
                data_year_df[['toleransi', 'kesetaraan', 'kerjasama']] = data_year_df[['toleransi', 'kesetaraan', 'kerjasama']].apply(pd.to_numeric) # Data diubah menjadi bentuk numerik
                
                # Data di grouping berdasarkan provinsinya dengan komponen yang diambil toleransi, kesetaraan dan kerjasama untuk dicluster
                grouped_data = data_year_df.groupby('provinsi').agg({
                    'toleransi': 'mean',
                    'kesetaraan': 'mean',
                    'kerjasama': 'mean'
                }).reset_index()

                imputer = SimpleImputer(strategy='mean') # Di imputasi apabila terdapat missing value menggunakan mean
                X = imputer.fit_transform(grouped_data[['toleransi', 'kesetaraan', 'kerjasama']]) # Pada ketiga kolom tersebut
                scaler = StandardScaler() # Menstandarisasi fitur dengan mengubah mean menjadi variance
                X_scaled = scaler.fit_transform(X) # Data yang sudah diimputasi kemudian distandarisasi

                # Menginisialisasi model GMM
                gmm = GaussianMixture(n_components=selected_clusters, random_state=42) # Random state 42 sudah menjadi standar karena menghasilkan data yang konsisten
                gmm.fit(X_scaled)
                labels = gmm.predict(X_scaled)

                labels = labels.astype(int)  

                grouped_data['cluster'] = labels
                grouped_data = self.reassign_clusters(grouped_data, selected_clusters)
                grouped_data[['toleransi', 'kesetaraan', 'kerjasama']] = grouped_data[['toleransi', 'kesetaraan', 'kerjasama']].applymap(lambda x: f"{x:.3f}") # Menampilkan 3 digit di belakang koma .3f

                # Generate plot GMM dengan ukuran 15x10
                fig = plt.figure(figsize=(15, 10))
                ax = fig.add_subplot(111, projection='3d')

                scatter = ax.scatter(grouped_data['toleransi'].astype(float), grouped_data['kesetaraan'].astype(float), grouped_data['kerjasama'].astype(float), c=grouped_data['cluster'], cmap='Set1')

                ax.set_title(f'3D Clusters for year {year}')
                ax.set_xlabel('Toleransi')
                ax.set_ylabel('Kesetaraan')
                ax.set_zlabel('Kerjasama')
                legend = ax.legend(*scatter.legend_elements(), title='Cluster')
                ax.add_artist(legend)

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

    # Data cluster di reassign berdasarkan urutan dari yang terendah
    def reassign_clusters(self, data, n_clusters):
        cluster_means = data.groupby('cluster')['toleransi'].mean().sort_values().reset_index()
        cluster_means['new_cluster'] = range(1, n_clusters + 1)
        cluster_mapping = cluster_means.set_index('cluster')['new_cluster'].to_dict()
        data['cluster'] = data['cluster'].map(cluster_mapping)
        return data


class ExportCSV(View):
    def get(self, request, *args, **kwargs):
        selected_year = request.GET.get('year', 2021)
        selected_clusters = int(request.GET.get('n_clusters', 8))

        gmm_data = self.fetch_gmm_data(selected_year, selected_clusters)

        if isinstance(gmm_data, pd.DataFrame) and not gmm_data.empty:
            # Membuat CSV
            response = HttpResponse(content_type='text/csv')
            response['Content-Disposition'] = f'attachment; filename="gmm_clusters_{selected_year}.csv"'

            # Menuliskan row CSV dengan header dibawah
            writer = csv.writer(response)
            writer.writerow(['No.', 'Year', 'Provinsi', 'Toleransi', 'Kesetaraan', 'Kerjasama', 'Cluster'])

            for idx, row in gmm_data.iterrows():
                writer.writerow([idx + 1, selected_year, row['provinsi'], row['toleransi'], row['kesetaraan'], row['kerjasama'], row['cluster']])

            return response

        else:
            return HttpResponse("No data available for the selected year and clusters.")

    # Data di fetch dari hasil perhitungan
    def fetch_gmm_data(self, selected_year, selected_clusters):
        data_year = Kub.objects.filter(tahun=selected_year)

        if data_year.exists():
            data_year_df = pd.DataFrame(list(data_year.values_list('id', 'provinsi', 'toleransi', 'kesetaraan', 'kerjasama')), columns=['id', 'provinsi', 'toleransi', 'kesetaraan', 'kerjasama'])
            data_year_df[['toleransi', 'kesetaraan', 'kerjasama']] = data_year_df[['toleransi', 'kesetaraan', 'kerjasama']].apply(pd.to_numeric)

            grouped_data = data_year_df.groupby('provinsi').agg({
                'toleransi': 'mean',
                'kesetaraan': 'mean',
                'kerjasama': 'mean'
            }).reset_index()

            imputer = SimpleImputer(strategy='mean')
            X = imputer.fit_transform(grouped_data[['toleransi', 'kesetaraan', 'kerjasama']])
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            gmm = GaussianMixture(n_components=selected_clusters, random_state=42)
            gmm.fit(X_scaled)
            labels = gmm.predict(X_scaled)

            grouped_data['cluster'] = labels
            grouped_data = self.reassign_clusters(grouped_data, selected_clusters)
            grouped_data[['toleransi', 'kesetaraan', 'kerjasama']] = grouped_data[['toleransi', 'kesetaraan', 'kerjasama']].applymap(lambda x: f"{x:.3f}")
            
            return grouped_data
        
        else:
            return pd.DataFrame()

    def reassign_clusters(self, grouped_data, selected_clusters):
        cluster_means = grouped_data.groupby('cluster')['toleransi'].mean().sort_values().reset_index()
        cluster_means['new_cluster'] = range(1, selected_clusters + 1)
        cluster_mapping = cluster_means.set_index('cluster')['new_cluster'].to_dict()
        grouped_data['cluster'] = grouped_data['cluster'].map(cluster_mapping)
        return grouped_data

# Halaman preprocessing
class Preprocessing(View):
    def get(self, request, *args, **kwargs):
        search_query = request.GET.get('search', '')

        # Data mana aja yang bisa di search di halaman preprocessing
        records = Kub.objects.filter(
            Q(provinsi__icontains=search_query) |
            Q(kabupaten__icontains=search_query) |
            Q(agama__icontains=search_query) |
            Q(tahun__icontains=search_query)
        ).order_by('id')

        # Paginate data
        paginator = Paginator(records, 10)  # 10 data perhalaman
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
            "page": paginated_records.number
        })

    def post(self, request, *args, **kwargs):
        if request.method == "POST":
            # Simpan data ke dalam server
            kub_file = request.FILES.get("kub_file")
            if kub_file:
                filename = kub_file.name
                with open(filename, 'wb+') as destination:
                    for chunk in kub_file.chunks():
                        destination.write(chunk)

                # Reset row dari 0 untuk menghitung jumlah row yang diupload
                row_count = 0
                form_errors = {}

                with open(filename, 'r') as cleaned_csv:
                    csv_reader = reader(cleaned_csv)
                    next(csv_reader)

                    # Menangani jumlah kolom agar harus sesuai 13 kolom
                    for row in csv_reader:
                        row_data = row[0].split(',')
                        if len(row_data) != 13:
                            print(f"Row has incorrect number of values: {row_data}")
                            continue
                        
                        # Kolom yang diunggah harus sesuai dengan ini
                        provinsi, kabupaten, jenis_kelamin, usia, agama, menikah, pendidikan, pekerjaan, kub, toleransi, kesetaraan, kerjasama, tahun = row_data

                        # Handle nilai NA menjadi 0
                        jenis_kelamin = int(jenis_kelamin) if jenis_kelamin != 'NA' else 0
                        usia = int(usia) if usia != 'NA' else 0
                        menikah = int(menikah) if menikah != 'NA' else 0
                        pendidikan = int(pendidikan) if pendidikan != 'NA' else 0
                        pekerjaan = int(pekerjaan) if pekerjaan != 'NA' else 0
                        kub = float(kub) if kub != 'NA' else 0.0
                        toleransi = float(toleransi) if toleransi != 'NA' else 0.0
                        kesetaraan = float(kesetaraan) if kesetaraan != 'NA' else 0.0
                        kerjasama = float(kerjasama) if kerjasama != 'NA' else 0.0
                        tahun = int(tahun)

                        # Data yang tidak sesuai dalam range akan di skip
                        if (jenis_kelamin not in [1, 2]) or (menikah not in range(1, 5)) or (tahun not in range(2021, 2024)):
                            continue

                        # Handle missing values
                        if pd.isna(provinsi):
                            provinsi = cleaned_csv.provinsi.mean()
                        if pd.isna(kabupaten):
                            kabupaten = cleaned_csv.kabupaten.mean()
                        if pd.isna(pendidikan):
                            pendidikan = (pendidikan % 13) if (pendidikan % 13 != 0) else 12

                        # Simpan data
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
                            row_count += 1
                        else:
                            form_errors[row_count + 1] = form.errors

                return render(request, 'pages/preprocessing.html', {
                    "form": UploadForm(),
                    "row_count": row_count,
                    "form_errors": form_errors,
                    "segment": "preprocessing"
                })
            else:
                return render(request, 'pages/preprocessing.html', {
                    "form": UploadForm(),
                    "file_missing": True,
                    "segment": "preprocessing"
                })
        else:
            return render(request, 'pages/preprocessing.html', {
                "form": UploadForm(),
                "segment": "preprocessing"
            })

# Halaman visualisasi
class Visualize(View):
    def get(self, request, *args, **kwargs):
        upload_form = UploadClusterForm()
        clusters = ClusterModel.objects.all().order_by('year')

        paginator = Paginator(clusters, 34)  # Menampilkan 34 data per halaman

        page = request.GET.get('page')
        try:
            clusters = paginator.page(page)
        except PageNotAnInteger:
            clusters = paginator.page(1)
        except EmptyPage:
            clusters = paginator.page(paginator.num_pages)

        return render(request, 'pages/visualize.html', {
            'segment': 'visualize',
            'upload_form': upload_form,
            'clusters': clusters,
        })
    
    # Upload file hasil clustering
    def post(self, request, *args, **kwargs):
        upload_form = UploadClusterForm(request.POST, request.FILES)
        if upload_form.is_valid():
            file = request.FILES['cluster_file']
            decoded_file = file.read().decode('utf-8').splitlines()
            reader = csv.DictReader(decoded_file)
            
            # Mapping data agar sesuai nama provinsinya dengan nama provinsi di geojson
            province_mapping = {
                "Aceh": "Aceh",
                "Bali": "Bali",
                "Bangka Belitung": "Bangka-Belitung",
                "Banten": "Banten",
                "Bengkulu": "Bengkulu",
                "DI Yogyakarta": "Yogyakarta",
                "DKI Jakarta": "Jakarta Raya",
                "Gorontalo": "Gorontalo",
                "Jambi": "Jambi",
                "Jawa Barat": "Jawa Barat",
                "Jawa Tengah": "Jawa Tengah",
                "Jawa Timur": "Jawa Timur",
                "Kalimantan Barat": "Kalimantan Barat",
                "Kalimantan Selatan": "Kalimantan Selatan",
                "Kalimantan Tengah": "Kalimantan Tengah",
                "Kalimantan Timur": "Kalimantan Timur",
                "Kalimantan Utara": "Kalimantan Utara",
                "Kep. Riau": "Kepulauan Riau",
                "Lampung": "Lampung",
                "Maluku Utara": "Maluku Utara",
                "Maluku": "Maluku",
                "NTB": "Nusa Tenggara Barat",
                "NTT": "Nusa Tenggara Timur",
                "Papua Barat": "Papua Barat",
                "Papua": "Papua",
                "Riau": "Riau",
                "Sulawesi Barat": "Sulawesi Barat",
                "Sulawesi Selatan": "Sulawesi Selatan",
                "Sulawesi Tengah": "Sulawesi Tengah",
                "Sulawesi Tenggara": "Sulawesi Tenggara",
                "Sulawesi Utara": "Sulawesi Utara",
                "Sumatera Barat": "Sumatera Barat",
                "Sumatera Selatan": "Sumatera Selatan",
                "Sumatera Utara": "Sumatera Utara"
            }

            for row in reader:
                province_name = province_mapping.get(row['Provinsi'].strip('"'), None)
                if province_name:
                    province, created = Province.objects.get_or_create(name=province_name)
                    ClusterModel.objects.create(
                        province=province,
                        year=int(row['Year']),
                        cluster=int(row['Cluster']),
                        toleransi=float(row['Toleransi']),
                        kesetaraan=float(row['Kesetaraan']),
                        kerjasama=float(row['Kerjasama']),
                    )
            return redirect('visualize')
        
        clusters = ClusterModel.objects.all()
        paginator = Paginator(clusters, 34)

        page = request.GET.get('page')
        try:
            clusters = paginator.page(page)
        except PageNotAnInteger:
            clusters = paginator.page(1)
        except EmptyPage:
            clusters = paginator.page(paginator.num_pages)
            
        return render(request, 'pages/visualize.html', {
            'segment': 'visualize',
            'upload_form': upload_form,
            'clusters': clusters,
        })

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