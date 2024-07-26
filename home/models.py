from django.db import models

# Create your models here.
class Kub(models.Model):
    provinsi = models.CharField(max_length=250)
    kabupaten = models.CharField(max_length=250)
    jenis_kelamin = models.IntegerField()
    usia = models.IntegerField()
    agama = models.CharField(max_length=100)
    menikah = models.IntegerField()
    pendidikan = models.IntegerField()
    pekerjaan = models.IntegerField()
    kub = models.FloatField()
    toleransi = models.FloatField()
    kesetaraan = models.FloatField()
    kerjasama = models.FloatField()
    tahun = models.IntegerField()

class Province(models.Model):
    name = models.CharField(max_length=100)

    def __str__(self):
        return self.name

class ClusterModel(models.Model):
    province = models.ForeignKey(Province, on_delete=models.CASCADE)
    year = models.IntegerField()
    cluster = models.IntegerField()
    toleransi = models.FloatField(null=True)
    kesetaraan = models.FloatField(null=True)
    kerjasama = models.FloatField(null=True)

    def __str__(self):
        return f"{self.province.name} - {self.year}: Cluster {self.cluster}"