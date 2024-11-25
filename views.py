import os
import csv
import cv2
import numpy as np
from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from .models import DiseaseHistory
from django.core.serializers.json import DjangoJSONEncoder
import random
import json
import calendar



MODEL_PATH = os.path.join(os.path.dirname(__file__), 'LettuceModel.h5')
model = load_model(MODEL_PATH)

CLASS_LABELS = {
    0: 'Bacterial',
    1: 'Fungal',
    2: 'Healthy',
}

RECOMMENDATIONS = {
    'Bacterial': [
        'Apply copper-based fungicides and improve air circulation.',
        'Increase water drainage and remove affected leaves.',
        'Use bactericides like streptomycin or copper hydroxide.',
        'Increase soil aeration to reduce bacterial spread.',
        'Apply beneficial bacteria to outcompete the harmful bacteria.',
        'Avoid overhead irrigation and water early in the day.',
        'Sanitize tools and equipment to prevent contamination.',
        'Remove infected plants to limit the spread of bacteria.',
        'Improve plant spacing to reduce humidity around plants.',
        'Monitor and control insect pests that may spread bacteria.'
    ],
    'Fungal': [
        'Use fungicides containing metalaxyl or fosetyl-aluminum.',
        'Increase airflow around plants to reduce humidity.',
        'Remove and destroy infected plant material to prevent spread.',
        'Consider applying sulfur-based fungicides.',
        'Ensure proper spacing between plants for airflow.',
        'Avoid watering plants from above, as wet foliage encourages fungal growth.',
        'Monitor soil pH levels to prevent favorable conditions for fungi.',
        'Install drip irrigation to avoid wetting plant leaves.',
        'Use resistant plant varieties to reduce the impact of fungal infections.',
        'Rotate crops annually to avoid re-infecting plants.'
    ],
    'Healthy': [
        'No action needed, continue regular maintenance.',
        'Ensure proper irrigation and fertilization for optimal growth.',
        'Prune dead or yellowing leaves to maintain plant health.',
        'Maintain healthy soil by adding organic compost.',
        'Ensure adequate sunlight for healthy plant development.',
        'Monitor humidity and temperature levels regularly.',
        'Control pests to prevent future infections.',
        'Maintain good plant spacing to avoid overcrowding.',
        'Much around plants to retain moisture and prevent weeds.',
        'Regularly inspect plants for early signs of disease.'
    ]
}

def home_view(request):
    return render(request, 'predictions/home.html')

def visualization_view(request):
    history = DiseaseHistory.objects.values('predicted_class', 'timestamp', 'probability')

    monthly_data = {month: {'Healthy': 0, 'Bacterial': 0, 'Fungal': 0} for month in calendar.month_name if month}
    monthly_counts = {month: {'Healthy': 0, 'Bacterial': 0, 'Fungal': 0} for month in calendar.month_name if month}

    for entry in history:
        predicted_class = entry['predicted_class']
        timestamp = entry['timestamp']
        probability = entry['probability']

        month = calendar.month_name[timestamp.month]
        
        if predicted_class in monthly_data[month]:
            monthly_data[month][predicted_class] += probability
            monthly_counts[month][predicted_class] += 1

    averaged_data = {
        month: {
            cls: (monthly_data[month][cls] / monthly_counts[month][cls]) if monthly_counts[month][cls] > 0 else 0
            for cls in monthly_data[month]
        }
        for month in monthly_data
    }

    data_for_chart = {
        'months': list(averaged_data.keys()),
        'healthy': [averaged_data[month]['Healthy'] for month in averaged_data],
        'bacterial': [averaged_data[month]['Bacterial'] for month in averaged_data],
        'fungal': [averaged_data[month]['Fungal'] for month in averaged_data],
    }
    history_json = json.dumps(data_for_chart, cls=DjangoJSONEncoder)

    return render(request, 'predictions/visualization.html', {'history_json': history_json})


def predict_view(request):
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_image = request.FILES['image']
        fs = FileSystemStorage()
        image_path = fs.save(uploaded_image.name, uploaded_image)
        full_path = fs.path(image_path)

        img = load_img(full_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)[0]
        predicted_index = np.argmax(predictions)
        predicted_class = CLASS_LABELS[predicted_index]
        probability = predictions[predicted_index] * 100

        recommendation = random.choice(RECOMMENDATIONS[predicted_class])

        DiseaseHistory.objects.create(
            image=uploaded_image,
            predicted_class=predicted_class,
            probability=probability,
            recommendation=recommendation
        )

        return render(request, 'predictions/result.html', {
            'predicted_class': predicted_class,
            'probability': f"{probability:.2f}%",
            'recommendation': recommendation,
            'image_url': fs.url(image_path)
        })
    return render(request, 'predictions/upload.html')

def camera_predict(request):
    if request.method == 'POST' and request.FILES.get('camera_image'):
        uploaded_image = request.FILES['camera_image']
        fs = FileSystemStorage()
        image_path = fs.save(uploaded_image.name, uploaded_image)
        full_path = fs.path(image_path)

        img = cv2.imread(full_path)
        img_resized = cv2.resize(img, (224, 224))
        img_array = img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)[0]
        predicted_index = np.argmax(predictions)
        predicted_class = CLASS_LABELS[predicted_index]
        probability = predictions[predicted_index] * 100

        recommendation = random.choice(RECOMMENDATIONS[predicted_class])

        DiseaseHistory.objects.create(
            image=uploaded_image,
            predicted_class=predicted_class,
            probability=probability,
            recommendation=recommendation
        )

        return render(request, 'predictions/camera.html', {
            'predicted_class': predicted_class,
            'probability': f"{probability:.2f}%",
            'recommendation': recommendation
        })

    return render(request, 'predictions/camera.html')

def download_csv(request):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="LettuceCare_History.csv"'
    writer = csv.writer(response)
    writer.writerow(['Image', 'Predicted Class', 'Probability', 'Recommendation', 'Timestamp'])
    for entry in DiseaseHistory.objects.all():
        writer.writerow([entry.image.url, entry.predicted_class, f"{entry.probability:.2f}%", entry.recommendation, entry.timestamp])
    return response

def download_pdf(request):
    # Create HttpResponse object with PDF headers
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="LettuceCare_History.pdf"'

    # Create a SimpleDocTemplate for the PDF
    doc = SimpleDocTemplate(response, pagesize=letter)
    elements = []

    # Title of the PDF
    styles = getSampleStyleSheet()
    title = Paragraph("Disease Prediction Report", styles['Title'])
    elements.append(title)
    elements.append(Spacer(1, 20))  # Add some space below the title

    # Prepare data for the table
    table_data = [['Type', 'Probability (%)', 'Timestamp']]  # Header row
    history_records = DiseaseHistory.objects.all()

    if history_records.exists():
        for entry in history_records:
            table_data.append([
                entry.predicted_class,
                f"{entry.probability:.2f}",
                entry.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            ])

        # Create the table
        table = Table(table_data, colWidths=[150, 150, 200])  
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.green),  
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),   
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),          
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),              
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),         
            ('GRID', (0, 0), (-1, -1), 1, colors.black),    
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),  
            ('FONTSIZE', (0, 1), (-1, -1), 10),             
        ]))

        elements.append(table)
    else:
        no_records_message = Paragraph("No disease prediction records found.", styles['BodyText'])
        elements.append(no_records_message)

    doc.build(elements)

    return response
def history_view(request):
    history = DiseaseHistory.objects.all().order_by('-timestamp')
    return render(request, 'predictions/history.html', {'history': history})

def evaluate_model(request):
    test_data_dir = 'SplitDataset_Lettuce/test'
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    true_labels = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())
    predictions = model.predict(test_generator)
    predicted_labels = np.argmax(predictions, axis=1)

    class_report = classification_report(true_labels, predicted_labels, target_names=class_labels, output_dict=True)
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    return render(request, 'predictions/evaluate.html', {
        'class_report': class_report,
        'conf_matrix': conf_matrix,
        'class_labels': class_labels
    })
