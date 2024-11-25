import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

MODEL_PATH = 'predictions/LettuceModel.h5'
model = load_model(MODEL_PATH)

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

print("Classification Report:")
print(classification_report(true_labels, predicted_labels, target_names=class_labels))

print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(true_labels, predicted_labels)
print(conf_matrix)
