from django.db import models

class DiseaseHistory(models.Model):
    image = models.ImageField(upload_to='uploads/')
    predicted_class = models.CharField(max_length=50)
    probability = models.FloatField()
    recommendation = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.predicted_class} - {self.timestamp}"
