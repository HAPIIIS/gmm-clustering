from django.urls import reverse  # Add this import for reverse lookup
from .models import Kub, ClusterModel, Province
from django.test import TestCase
from django.contrib.auth.models import User
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.metrics import silhouette_score

class ClusterViewTests(TestCase):
    def setUp(self):
        # Create a Province instance first
        self.province = Province.objects.create(name="Test Province")
        
        # Setup test data and user
        self.user = User.objects.create_user(username='testuser', password='testpassword')
        self.client.login(username='testuser', password='testpassword')
        
        # Create ClusterModel instance with the Province instance
        ClusterModel.objects.create(province=self.province, year=2021, cluster=1)

    def test_cluster_geojson_view(self):
        # Test the cluster_geojson view
        response = self.client.get(reverse('cluster_geojson'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Test Province")


class KubModelTests(TestCase):
    def setUp(self):
        # Setup test data for Kub model
        Kub.objects.create(
            provinsi="Test Province", 
            tahun=2021, toleransi=0.5, 
            kesetaraan=0.6, 
            kerjasama=0.7, 
            jenis_kelamin=1, 
            usia=25,  # Set a valid value for the usia field
            agama="islam", 
            menikah=1, pendidikan=10, 
            pekerjaan=1,
            kub=0.6
        )
    
    def test_kub_creation(self):
        # Test Kub model creation
        kub = Kub.objects.get(provinsi="Test Province")
        self.assertEqual(kub.toleransi, 0.5)
        self.assertEqual(kub.kesetaraan, 0.6)
        self.assertEqual(kub.kerjasama, 0.7)
        self.assertEqual(kub.jenis_kelamin, 1)
        self.assertEqual(kub.usia, 25)
        self.assertEqual(kub.agama, "islam")
        self.assertEqual(kub.menikah, 1)
        self.assertEqual(kub.pendidikan, 10)
        self.assertEqual(kub.pekerjaan, 1)
        self.assertEqual(kub.kub, 0.6)

    def test_kub_update(self):
        kub = Kub.objects.get(provinsi="Test Province")
        kub.toleransi = 0.8
        kub.save()
        self.assertEqual(kub.toleransi, 0.8)

    def test_kub_delete(self):
        kub = Kub.objects.get(provinsi="Test Province")
        kub.delete()
        self.assertEqual(Kub.objects.count(), 0)

class KubEdgeCaseTests(TestCase):

    def test_kub_creation_with_boundary_values(self):
        kub = Kub.objects.create(
            provinsi="Test Province", 
            tahun=2021, 
            toleransi=0.0, 
            kesetaraan=1.0, 
            kerjasama=0.7, 
            jenis_kelamin=1, 
            usia=0,  
            agama="islam", 
            menikah=1, 
            pendidikan=10, 
            pekerjaan=1,
            kub=0.6
        )
        self.assertEqual(kub.toleransi, 0.0)
        self.assertEqual(kub.kesetaraan, 1.0)
        self.assertEqual(kub.usia, 0)

class GMMClusteringTest(TestCase):
    def setUp(self):
        # Setup data for testing
        self.data_2021 = Kub.objects.filter(tahun=2021)
        self.data_2022 = Kub.objects.filter(tahun=2022)
        self.data_2023 = Kub.objects.filter(tahun=2023)

    def test_gmm_clustering(self):
        for year in range(2021, 2024):  # Iterate over the years 2021 to 2023
            data_year = Kub.objects.filter(tahun=year)

            if data_year.exists():
                features = ['toleransi', 'kesetaraan', 'kerjasama']
                X = [[getattr(item, feature) for feature in features] for item in data_year]
                data_year_df = pd.DataFrame(X, columns=features)
                imputer = SimpleImputer(strategy='mean')
                X = imputer.fit_transform(data_year_df)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                gmm = GaussianMixture(n_components=3, random_state=42)
                gmm.fit(X_scaled)
                labels = gmm.predict(X_scaled)
                
                # Verify the clusters
                self.assertEqual(len(set(labels)), 3)

class GMMClusterEvaluationTest(TestCase):
    def setUp(self):
        # Setup data for testing
        self.data_2021 = Kub.objects.filter(tahun=2021)
        self.data_2022 = Kub.objects.filter(tahun=2022)
        self.data_2023 = Kub.objects.filter(tahun=2023)

    def test_cluster_evaluation(self):
        for year in range(2021, 2024): 
            data_year = Kub.objects.filter(tahun=year)

            if data_year.exists():
                features = ['toleransi', 'kesetaraan', 'kerjasama']
                X = [[getattr(item, feature) for feature in features] for item in data_year]
                data_year_df = pd.DataFrame(X, columns=features)
                imputer = SimpleImputer(strategy='mean')
                X = imputer.fit_transform(data_year_df)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                gmm = GaussianMixture(n_components=2, random_state=42)
                gmm.fit(X_scaled)
                labels = gmm.predict(X_scaled)
                score = silhouette_score(X_scaled, labels)

                self.assertGreater(score, 0.5)

