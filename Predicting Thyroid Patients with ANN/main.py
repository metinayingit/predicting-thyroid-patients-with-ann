import tkinter as tk
from tkinter import messagebox
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss, precision_score, recall_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# File paths of training and testing datasets
train_file_path = 'C:/Users/metinayin/Desktop/GitHub/predicting-thyroid-patients-with-ann/Predicting Thyroid Patients with ANN/ann-train.data'
test_file_path = 'C:/Users/metinayin/Desktop/GitHub/predicting-thyroid-patients-with-ann/Predicting Thyroid Patients with ANN/ann-test.data'

# Function that loads the data set
def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            values = line.strip().split()
            if len(values) == 22 and all(v != '' for v in values):
                data.append(list(map(float, values)))
    return pd.DataFrame(data)

# Loading training and testing datasets
train_df = load_data(train_file_path)
test_df = load_data(test_file_path)

# Preparation of sample data from each class
example_normal = train_df[train_df.iloc[:, -1] == 1].head(2).values.tolist()
example_hyperthyroid = train_df[train_df.iloc[:, -1] == 2].head(2).values.tolist()
example_hypothyroid = train_df[train_df.iloc[:, -1] == 3].head(2).values.tolist()

# Separation of training and testing datasets into features (X) and labels (y)
X_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]

X_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]

# Scaling datasets to standard normal distribution
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Creating and training the MLP classifier model
mlp = MLPClassifier(hidden_layer_sizes=(30, 30, 30), max_iter=100)
mlp.fit(X_train, y_train)

# Calculation of model predictions and evaluation metrics
predictions = mlp.predict(X_test)
cm = confusion_matrix(y_test, predictions)
cr = classification_report(y_test, predictions, output_dict=True)
accuracy = accuracy_score(y_test, predictions)
loss_value = log_loss(y_test, mlp.predict_proba(X_test))

# Function that takes data from the user and makes predictions
def predict():
    try:
        user_data = [float(entry.get()) for entry in entries]
        if len(user_data) != 21:
            raise ValueError("Missing entry")
        user_data = scaler.transform([user_data])
        prediction = mlp.predict(user_data)[0]
        result = {1: 'Normal', 2: 'Hyperthyroidi', 3: 'Hypothyroid'}
        result_label.config(text=f'Prediction: {result[prediction]}', fg='green' if prediction == 1 else 'orange' if prediction == 2 else 'red')
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numbers in all fields.")

# Function that fills sample data into input fields
def fill_example(example):
    for i, value in enumerate(example):
        entries[i].delete(0, tk.END)
        entries[i].insert(0, str(value))

# Function that prepares the classification report as text
def display_classification_report(cr):
    report_text = f'''
    1. Class (Normal):
    - Precision: {cr["1.0"]["precision"]:.2f}
    - Recall: {cr["1.0"]["recall"]:.2f}
    - F1 Score: {cr["1.0"]["f1-score"]:.2f}
    - Support: {cr["1.0"]["support"]}
    
    2. Class (Hyperthyroid):
    - Precision: {cr["2.0"]["precision"]:.2f}
    - Recall: {cr["2.0"]["recall"]:.2f}
    - F1 Score: {cr["2.0"]["f1-score"]:.2f}
    - Support: {cr["2.0"]["support"]}
    
    3. Class (Hypothyroid):
    - Precision: {cr["3.0"]["precision"]:.2f}
    - Recall: {cr["3.0"]["recall"]:.2f}
    - F1 Score: {cr["3.0"]["f1-score"]:.2f}
    - Support: {cr["3.0"]["support"]}
    
    Overall:
    - Accuracy: {accuracy:.2f}
    - Precision (macro avg): {cr["macro avg"]["precision"]:.2f}
    - Recall (macro avg): {cr["macro avg"]["recall"]:.2f}
    - F1 Score (macro avg): {cr["macro avg"]["f1-score"]:.2f}
    - Precision (weighted avg): {cr["weighted avg"]["precision"]:.2f}
    - Recall (weighted avg): {cr["weighted avg"]["recall"]:.2f}
    - F1 Score (weighted avg): {cr["weighted avg"]["f1-score"]:.2f}
    - Loss Value: {loss_value:.2f}
    '''
    return report_text

# Function that visualizes the confusion matrix
def plot_confusion_matrix(cm):
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap='Oranges')
    fig.colorbar(cax)
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f'{val}', ha='center', va='center')
    plt.xlabel('Estimated')
    plt.ylabel('Real')
    plt.title('Confusion Matrix')
    return fig

# Function that draws the loss curve
def plot_loss_curve():
    fig, ax = plt.subplots()
    ax.plot(mlp.loss_curve_)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss Value')
    ax.set_title('Loss Value Curve')
    return fig

# Function that runs the model 25 times and analyzes the results
def analyze_model():
    results = {'accuracy': [], 'precision': [], 'recall': [], 'log_loss': []}
    n_runs = 25
    for _ in range(n_runs):
        mlp = MLPClassifier(hidden_layer_sizes=(30, 30, 30), max_iter=100, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10)
        mlp.fit(X_train, y_train)
        predictions = mlp.predict(X_test)
        probas = mlp.predict_proba(X_test)
        results['accuracy'].append(accuracy_score(y_test, predictions))
        results['precision'].append(precision_score(y_test, predictions, average='macro'))
        results['recall'].append(recall_score(y_test, predictions, average='macro'))
        results['log_loss'].append(log_loss(y_test, probas))
    mean_results = pd.DataFrame(results).mean()
    std_results = pd.DataFrame(results).std()
    analysis_text = f'''
    The model was run 25 times:
    - Average Accuracy: {mean_results['accuracy']:.2f} (± {std_results['accuracy']:.2f})
    - Average Precision: {mean_results['precision']:.2f} (± {std_results['precision']:.2f})
    - Average Recall: {mean_results['recall']:.2f} (± {std_results['recall']:.2f})
    - Average Log Loss: {mean_results['log_loss']:.2f} (± {std_results['log_loss']:.2f})
    '''
    analysis_label.config(text=analysis_text)

# Calculation of class distributions in training and test data sets
train_class_counts = train_df.iloc[:, -1].value_counts()
test_class_counts = test_df.iloc[:, -1].value_counts()
attribute_analysis = train_df.describe().loc[['mean', 'std']]
attribute_analysis_text = "\n".join(f"{attr}:\n  Average: {values['mean']:.2f}\n  Std deviation: {values['std']:.2f}\n" for attr, values in attribute_analysis.iteritems())

# Creating the main window
window = tk.Tk()
window.title("Thyroid Disease Prediction with Artificial Neural Networks")
window.geometry('1600x960')

# Creating and placing a main frame
frame = tk.Frame(window)
frame.pack(padx=10, pady=10, fill='both', expand=True)

# Frame in which graphics will be displayed
graphics_frame = tk.LabelFrame(frame, text="Graphics", font=('Arial', 12, 'bold'))
graphics_frame.grid(row=0, column=0, padx=10, pady=10, sticky='ns')

# Plotting the confusion matrix
canvas1 = FigureCanvasTkAgg(plot_confusion_matrix(cm), master=graphics_frame)
canvas1.draw()
canvas1.get_tk_widget().pack(padx=5, pady=5)

# Drawing the loss curve
canvas2 = FigureCanvasTkAgg(plot_loss_curve(), master=graphics_frame)
canvas2.draw()
canvas2.get_tk_widget().pack(padx=5, pady=5)

# Frame showing classification report and accuracy values
report_frame = tk.LabelFrame(frame, text="Classification Report and Accuracy Values", font=('Arial', 12, 'bold'))
report_frame.grid(row=0, column=1, padx=10, pady=10, sticky='n')

# Printing the classification report on the screen
for line in display_classification_report(cr).split('\n'):
    tk.Label(report_frame, text=line, justify='left', anchor='w', font=('Arial', 12)).pack(anchor='w')

# Tooltip functions
def on_leave(event):
    if hasattr(event.widget, 'tooltip'):
        event.widget.tooltip.destroy()

def on_enter(event, attribute):
    text = attribute_descriptions.get(attribute, "No information.")
    tooltip = tk.Toplevel(window, bg="white")
    tooltip.wm_overrideredirect(True)
    tooltip.geometry(f"+{event.x_root+10}+{event.y_root+10}")
    tk.Label(tooltip, text=text, bg="white", font=('Arial', 10, 'normal'), wraplength=300).pack()
    event.widget.tooltip = tooltip

# Creating user login fields
input_frame = tk.LabelFrame(frame, text="Attributes", font=('Arial', 12, 'bold'))
input_frame.grid(row=0, column=2, padx=10, pady=10, sticky='n')

# Input fields and attribute descriptions
attributes = [
    ("Age", "0.01 - 0.94"), 
    ("Sex", "0: Male, 1: Female"), 
    ("On thyroxine", "0: No, 1: Yes"), 
    ("Query on thyroxine", "0: No, 1: Yes"), 
    ("On antithyroid medication", "0: No, 1: Yes"), 
    ("Sick", "0: No, 1: Yes"), 
    ("Pregnant", "0: No, 1: Yes"), 
    ("Thyroid surgery", "0: No, 1: Yes"), 
    ("I121 treatment", "0: No, 1: Yes"), 
    ("Query hypothyroid", "0: No, 1: Yes"), 
    ("Query hyperthyroid", "0: No, 1: Yes"), 
    ("Lithium", "0: No, 1: Yes"), 
    ("Goitre", "0: No, 1: Yes"), 
    ("Tumor", "0: No, 1: Yes"), 
    ("Hypopituitary", "0: No, 1: Yes"), 
    ("TSH", "0.0005 - 0.1059"), 
    ("T3", "0.002 - 0.43"), 
    ("TT4", "0.019 - 0.232"), 
    ("T4U", "0.002 - 0.612"), 
    ("FTI", "1.0 - 3.0"),
    ("Referral source", "0 - 1")
]

attribute_descriptions = {
    "Age": "The patient's age.",
    "Sex": "The patient's gender.",
    "On thyroxine": "Is the patient on thyroxine?",
    "Query on thyroxine": "Is there a query on thyroxine treatment?",
    "On antithyroid medication": "Is the patient on antithyroid medication?",
    "Sick": "Is the patient sick?",
    "Pregnant": "Is the patient pregnant?",
    "Thyroid surgery": "Has the patient had thyroid surgery?",
    "I121 treatment": "Has the patient received I121 radioactive treatment?",
    "Query hypothyroid": "Preliminary assessment result.",
    "Query hyperthyroid": "Preliminary assessment result.",
    "Lithium": "Is the patient on lithium treatment?",
    "Goitre": "Does the patient have goitre?",
    "Tumor": "Does the patient have a tumor?",
    "Hypopituitary": "Does the patient have hypopituitarism?",
    "TSH": "Thyroid Stimulating Hormone level.",
    "T3": "Triiodothyronine level, T3 thyroid hormone level.",
    "TT4": "Total Thyroxine level.",
    "T4U": "Thyroxine Binding Capacity.",
    "FTI": "Free Thyroxine Index.",
    "Referral source": "The source from which the patient was referred."
}

# Creating and placing entry areas
entries = []
for i, (attribute, value_range) in enumerate(attributes):
    label = tk.Label(input_frame, text=f'{attribute} ({value_range}):', font=('Arial', 12))
    label.grid(row=i, column=0, padx=5, pady=2, sticky='e')
    entry = tk.Entry(input_frame)
    entry.grid(row=i, column=1, padx=5, pady=2)
    entries.append(entry)
    label.bind("<Enter>", lambda e, attribute=attribute: on_enter(e, attribute))
    label.bind("<Leave>", on_leave)

# Make a prediction button
tk.Button(input_frame, text="Prediction", command=predict, font=('Arial', 12, 'bold'), bg='#84d4f5').grid(row=len(attributes), column=0, columnspan=2, pady=10)

# Frame to display sample data
example_frame = tk.LabelFrame(input_frame, text="Example Datas", font=('Arial', 12, 'bold'))
example_frame.grid(row=len(attributes)+1, column=0, columnspan=2, pady=10)

# Creating and placing sample data buttons
example_buttons = [
    ("Normal Example 1", example_normal[0], '#b8ffb8'),
    ("Normal Example 2", example_normal[1], '#b8ffb8'),
    ("Hyperthyroid Example 1", example_hyperthyroid[0], '#ffb89c'),
    ("Hyperthyroid Example 2", example_hyperthyroid[1], '#ffb89c'),
    ("Hypothyroid Example 1", example_hypothyroid[0], '#f57171'),
    ("Hypothyroid Example 2", example_hypothyroid[1], '#f57171')
]

for i, (text, example, color) in enumerate(example_buttons):
    tk.Button(example_frame, text=text, command=lambda ex=example: fill_example(ex), font=('Arial', 12), bg=color).grid(row=i//2, column=i%2, padx=5, pady=2)

# Label where the prediction result will be displayed
result_label = tk.Label(input_frame, text="Prediction: ", font=('Arial', 12, 'bold'), fg='black')
result_label.grid(row=len(attributes)+2, column=0, columnspan=2, pady=10)

# Data set information framework
class_names = {1: "Normal", 2: "Hyperthyroid", 3: "Hypothyroid"}
dataset_info_frame = tk.LabelFrame(frame, text="Data Set Information", font=('Arial', 12, 'bold'))
dataset_info_frame.grid(row=0, column=3, padx=10, pady=10, sticky='n')

# Training dataset information
tk.Label(dataset_info_frame, text="Training Data Set Information:", font=('Arial', 12, 'bold')).pack(anchor='w', padx=5, pady=5)
for cls, count in train_df.iloc[:, -1].value_counts().sort_index().items():
    tk.Label(dataset_info_frame, text=f"Sınıf {int(cls)} ({class_names[int(cls)]}): {count} data", font=('Arial', 12)).pack(anchor='w', padx=20, pady=2)

# Test dataset information
tk.Label(dataset_info_frame, text="\nTest Data Set Information:", font=('Arial', 12, 'bold')).pack(anchor='w', padx=5, pady=5)
for cls, count in test_df.iloc[:, -1].value_counts().sort_index().items():
    tk.Label(dataset_info_frame, text=f"Sınıf {int(cls)} ({class_names[int(cls)]}): {count} data", font=('Arial', 12)).pack(anchor='w', padx=20, pady=2)

# Model analysis button and label to display the analysis
tk.Button(dataset_info_frame, text="Analysis", command=analyze_model, font=('Arial', 12, 'bold'), bg='#ffcccb').pack(pady=10)
analysis_label = tk.Label(dataset_info_frame, text="", font=('Arial', 12))
analysis_label.pack(pady=10)

# Main window loop
window.mainloop()