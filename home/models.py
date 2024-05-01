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