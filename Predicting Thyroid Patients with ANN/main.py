import tkinter as tk
from tkinter import messagebox
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss

# Modelin Eğitilmesi
train_file_path = 'C:/Users/metinayin/Desktop/GitHub/predicting-thyroid-patients-with-ann/Predicting Thyroid Patients with ANN/ann-train.data'
test_file_path = 'C:/Users/metinayin/Desktop/GitHub/predicting-thyroid-patients-with-ann/Predicting Thyroid Patients with ANN/ann-test.data'

train_data = []
with open(train_file_path, 'r') as file:
    for line in file:
        train_data.append(list(map(float, line.strip().split())))

test_data = []
with open(test_file_path, 'r') as file:
    for line in file:
        test_data.append(list(map(float, line.strip().split())))

train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

# Örnek verilerin seçilmesi (3 hasta, 3 sağlıklı)
example_hasta = train_df[train_df.iloc[:, -1] == 1].head(3).values.tolist()
example_saglikli = train_df[train_df.iloc[:, -1] == 3].head(3).values.tolist()

X_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]

X_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(30,30,30), max_iter=500)
mlp.fit(X_train, y_train)

predictions = mlp.predict(X_test)
cm = confusion_matrix(y_test, predictions)
cr = classification_report(y_test, predictions, output_dict=True)
accuracy = accuracy_score(y_test, predictions)
loss = log_loss(y_test, mlp.predict_proba(X_test))
print(cm)
print(classification_report(y_test, predictions))

# Tkinter Arayüzü
def predict():
    try:
        user_data = [float(entry.get()) for entry in entries]
        if len(user_data) != 21:
            raise ValueError("Eksik giriş")
        user_data = np.array(user_data).reshape(1, -1)
        user_data = scaler.transform(user_data)
        prediction = mlp.predict(user_data)
        if int(prediction[0]) == 1:
            result_label.config(text='Hasta', fg='red')
        else:
            result_label.config(text='Sağlıklı', fg='green')
    except ValueError:
        messagebox.showerror("Hata", "Lütfen tüm alanlara geçerli sayılar girin.")

def fill_example(example):
    for i, value in enumerate(example):
        entries[i].delete(0, tk.END)
        entries[i].insert(0, str(value))

def display_classification_report(cr):
    report_text = f'''
    \n1. Sınıf:\n- Kesinlik (precision): {cr["1.0"]["precision"]:.2f}\n- Duyarlılık (recall): {cr["1.0"]["recall"]:.2f}\n- F1 Skoru (f1-score): {cr["1.0"]["f1-score"]:.2f}\n- Destek (support): {cr["1.0"]["support"]}\n
    \n2. Sınıf:\n- Kesinlik (precision): {cr["2.0"]["precision"]:.2f}\n- Duyarlılık (recall): {cr["2.0"]["recall"]:.2f}\n- F1 Skoru (f1-score): {cr["2.0"]["f1-score"]:.2f}\n- Destek (support): {cr["2.0"]["support"]}\n
    \n3. Sınıf:\n- Kesinlik (precision): {cr["3.0"]["precision"]:.2f}\n- Duyarlılık (recall): {cr["3.0"]["recall"]:.2f}\n- F1 Skoru (f1-score): {cr["3.0"]["f1-score"]:.2f}\n- Destek (support): {cr["3.0"]["support"]}\n
    \nGenel:\n- Doğruluk (accuracy): {accuracy:.2f}\n- Kesinlik (macro avg): {cr["macro avg"]["precision"]:.2f}\n- Duyarlılık (macro avg): {cr["macro avg"]["recall"]:.2f}\n- F1 Skoru (macro avg): {cr["macro avg"]["f1-score"]:.2f}\n- Kesinlik (weighted avg): {cr["weighted avg"]["precision"]:.2f}\n- Duyarlılık (weighted avg): {cr["weighted avg"]["recall"]:.2f}\n- F1 Skoru (weighted avg): {cr["weighted avg"]["f1-score"]:.2f}\n- Loss Değeri: {loss:.2f}
    '''
    return report_text

window = tk.Tk()
window.title("Yapay Sinir Ağları ile Tiroid Hastalığı Tahmini")

frame = tk.Frame(window)
frame.pack(padx=10, pady=10)

# Sol tarafta sınıflandırma raporu ve doğruluk değerleri için çerçeve
report_frame = tk.LabelFrame(frame, text="Sınıflandırma Raporu ve Doğruluk Değerleri", font=('Arial', 11, 'bold'))
report_frame.grid(row=0, column=0, padx=10, pady=10, sticky='nw')

# Report text
report_text = display_classification_report(cr)
report_lines = report_text.split('\n')
for line in report_lines:
    if "Sınıf" in line or "Genel" in line:
        label = tk.Label(report_frame, text=line, justify='left', anchor='w', font=('Arial', 11, 'bold'))
    else:
        label = tk.Label(report_frame, text=line, justify='left', anchor='w', font=('Arial', 11))
    label.pack(anchor='w')

# Sağ tarafta veri girişleri için çerçeve
input_frame = tk.LabelFrame(frame, text="Veri Girişleri", font=('Arial', 11, 'bold'))
input_frame.grid(row=0, column=1, padx=10, pady=10, sticky='ne')

attributes = [
    ("Yaş", "0.01 - 0.94"), 
    ("Cinsiyet", "0: Erkek, 1: Kadın"), 
    ("Tiroksin kullanımı", "0: Hayır, 1: Evet"), 
    ("Tiroksin sorgulaması", "0: Hayır, 1: Evet"), 
    ("Antitiroid ilaç kullanımı", "0: Hayır, 1: Evet"), 
    ("Hasta", "0: Hayır, 1: Evet"), 
    ("Hamile", "0: Hayır, 1: Evet"), 
    ("Tiroit ameliyatı", "0: Hayır, 1: Evet"), 
    ("I131 tedavisi", "0: Hayır, 1: Evet"), 
    ("Hipotiroid şüphesi", "0: Hayır, 1: Evet"), 
    ("Hipertiroid şüphesi", "0: Hayır, 1: Evet"), 
    ("Lityum", "0: Hayır, 1: Evet"), 
    ("Guatr", "0: Hayır, 1: Evet"), 
    ("Tümör", "0: Hayır, 1: Evet"), 
    ("Hipopitüiter", "0: Hayır, 1: Evet"), 
    ("TSH değeri", "0.0005 - 0.1059"), 
    ("T3 değeri", "0.002 - 0.43"), 
    ("TT4 değeri", "0.019 - 0.232"), 
    ("T4U değeri", "0.002 - 0.612"), 
    ("FTI değeri", "1.0 - 3.0"),
    ("Yönlendirme kaynağı", "0 - 1")
]

entries = []
for i, (attribute, value_range) in enumerate(attributes):
    label = tk.Label(input_frame, text=f'{attribute} ({value_range}):', font=('Arial', 11))
    label.grid(row=i, column=0, padx=5, pady=2, sticky='e')
    entry = tk.Entry(input_frame)
    entry.grid(row=i, column=1, padx=5, pady=2)
    entries.append(entry)

predict_button = tk.Button(input_frame, text="Tahmin Et", command=predict, font=('Arial', 11, 'bold'))
predict_button.grid(row=len(attributes), column=0, columnspan=2, pady=10)

example_hasta_buttons = []
example_saglikli_buttons = []

for idx in range(3):
    example_hasta_buttons.append(tk.Button(input_frame, text=f"Örnek Hasta {idx+1}", command=lambda idx=idx: fill_example(example_hasta[idx]), font=('Arial', 11, 'bold')))
    example_hasta_buttons[idx].grid(row=len(attributes)+1+idx, column=0, pady=5)
    
    example_saglikli_buttons.append(tk.Button(input_frame, text=f"Örnek Sağlıklı {idx+1}", command=lambda idx=idx: fill_example(example_saglikli[idx]), font=('Arial', 11, 'bold')))
    example_saglikli_buttons[idx].grid(row=len(attributes)+1+idx, column=1, pady=5)

result_label = tk.Label(input_frame, text="Tahmin", font=('Arial', 13, 'bold'))
result_label.grid(row=len(attributes)+4, column=0, columnspan=2, pady=10)

window.mainloop()