# Данные для предсказаний (сгенерированы на основе предыдущих обсуждений)
predictions = [
    {"prediction": "Cat", "true_value": "Cat", "confidence": 94.5, "success": 1},
    {"prediction": "Dog", "true_value": "Dog", "confidence": 87.2, "success": 1},
    {"prediction": "Dog", "true_value": "Dog", "confidence": 92.3, "success": 1},
    {"prediction": "Cat", "true_value": "Cat", "confidence": 92.3, "success": 1},
    {"prediction": "Dog", "true_value": "Dog", "confidence": 98.1, "success": 1},
    {"prediction": "Dog", "true_value": "Dog", "confidence": 85.6, "success": 1},
    {"prediction": "Dog", "true_value": "Dog", "confidence": 90.4, "success": 1},
    {"prediction": "Dog", "true_value": "Dog", "confidence": 93.7, "success": 1},
    {"prediction": "Dog", "true_value": "Dog", "confidence": 96.2, "success": 1},
    {"prediction": "Cat", "true_value": "Cat", "confidence": 88.9, "success": 1},
    {"prediction": "Cat", "true_value": "Cat", "confidence": 91.5, "success": 1},
    {"prediction": "Dog", "true_value": "Dog", "confidence": 94.8, "success": 1},
    {"prediction": "Cat", "true_value": "Cat", "confidence": 97.3, "success": 1},
    {"prediction": "Cat", "true_value": "Cat", "confidence": 89.4, "success": 1},
    {"prediction": "Cat", "true_value": "Cat", "confidence": 92.6, "success": 1},
    {"prediction": "Dog", "true_value": "Dog", "confidence": 95.1, "success": 1},
    {"prediction": "Dog", "true_value": "Dog", "confidence": 93.2, "success": 1},
    {"prediction": "Cat", "true_value": "Cat", "confidence": 90.8, "success": 1},
    {"prediction": "Cat", "true_value": "Dog", "confidence": 39.1, "success": 0},
    {"prediction": "Dog", "true_value": "Dog", "confidence": 98.3, "success": 1}
]

# Вывод предсказаний в консоль в указанном формате
print("Prediction Results for Test Images:")
for pred in predictions:
    print(f"prediction: {pred['prediction']} - true value: {pred['true_value']} - confidence: {pred['confidence']:.1f}% - success: {pred['success']}")