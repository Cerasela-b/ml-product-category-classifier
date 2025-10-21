import joblib
import pandas as pd

# Load the saved model
model = joblib.load("model/category_classifier_model.pkl")

print("Model loaded successfully!")
print("Type 'exit' at any point to stop.\n")

while True:
    title = input("Enter product title: ")
    if title.lower() == "exit":
        print("Exiting...")
        break


    # Compute review length
    product_title_length = len(title)

    # Create a DataFrame from input
    user_input = pd.DataFrame([{
        "Product Title": title,
        "Product Title Length": product_title_length
    }])

    # Predict sentiment
    prediction = model.predict(user_input)[0]
    print(f"Predicted category: {prediction}\n" + "-" * 40)