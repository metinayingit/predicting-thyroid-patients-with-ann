import tkinter as tk
from tkinter import messagebox
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

train_file_path = 'C:/Users/metinayin/Desktop/GitHub/predicting-thyroid-patients-with-ann/Predicting Thyroid Patients with ANN/ann-train.data'
test_file_path = 'C:/Users/metinayin/Desktop/GitHub/predicting-thyroid-patients-with-ann/Predicting Thyroid Patients with ANN/ann-test.data'

def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            values = line.strip().split()
            if len(values) == 22 and all(v != '' for v in values):
                data.append(list(map(float, values)))
    return pd.DataFrame(data)

train_df = load_data(train_file_path)
test_df = load_data(test_file_path)

example_normal = train_df[train_df.iloc[:, -1] == 1].head(2).values.tolist()
example_hipertiroidi = train_df[train_df.iloc[:, -1] == 2].head(2).values.tolist()
example_hipotiroidi = train_df[train_df.iloc[:, -1] == 3].head(2).values.tolist()

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
loss_value = log_loss(y_test, mlp.predict_proba(X_test))

def predict():
    try:
        user_data = [float(entry.get()) for entry in entries]
        if len(user_data) != 21:
            raise ValueError("Eksik giriş")
        user_data = np.array(user_data).reshape(1, -1)
        user_data = scaler.transform(user_data)
        prediction = mlp.predict(user_data)
        if int(prediction[0]) == 1:
            result_label.config(text='Tahmin: Normal', fg='green')
        elif int(prediction[0]) == 2:
            result_label.config(text='Tahmin: Hipertiroidi', fg='orange')
        elif int(prediction[0]) == 3:
            result_label.config(text='Tahmin: Hipotiroidi', fg='red')
    except ValueError:
        messagebox.showerror("Hata", "Lütfen tüm alanlara geçerli sayılar girin.")

def fill_example(example):
    for i, value in enumerate(example):
        entries[i].delete(0, tk.END)
        entries[i].insert(0, str(value))

def display_classification_report(cr):
    report_text = f'''
    \n1. Sınıf (Normal):\n- Kesinlik (precision): {cr["1.0"]["precision"]:.2f}\n- Duyarlılık (recall): {cr["1.0"]["recall"]:.2f}\n- F1 Skoru (f1-score): {cr["1.0"]["f1-score"]:.2f}\n- Destek (support): {cr["1.0"]["support"]}\n
    \n2. Sınıf (Hipertiroidi):\n- Kesinlik (precision): {cr["2.0"]["precision"]:.2f}\n- Duyarlılık (recall): {cr["2.0"]["recall"]:.2f}\n- F1 Skoru (f1-score): {cr["2.0"]["f1-score"]:.2f}\n- Destek (support): {cr["2.0"]["support"]}\n
    \n3. Sınıf (Hipotiroidi):\n- Kesinlik (precision): {cr["3.0"]["precision"]:.2f}\n- Duyarlılık (recall): {cr["3.0"]["recall"]:.2f}\n- F1 Skoru (f1-score): {cr["3.0"]["f1-score"]:.2f}\n- Destek (support): {cr["3.0"]["support"]}\n
    \nGenel:\n- Doğruluk (accuracy): {accuracy:.2f}\n- Kesinlik (macro avg): {cr["macro avg"]["precision"]:.2f}\n- Duyarlılık (macro avg): {cr["macro avg"]["recall"]:.2f}\n- F1 Skoru (macro avg): {cr["macro avg"]["f1-score"]:.2f}\n- Kesinlik (weighted avg): {cr["weighted avg"]["precision"]:.2f}\n- Duyarlılık (weighted avg): {cr["weighted avg"]["recall"]:.2f}\n- F1 Skoru (weighted avg): {cr["weighted avg"]["f1-score"]:.2f}\n- Loss Değeri: {loss_value:.2f}
    '''
    return report_text

def plot_confusion_matrix(cm):
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap='Blues')
    fig.colorbar(cax)
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f'{val}', ha='center', va='center')
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek')
    plt.title('Confusion Matrix')
    return fig

def plot_loss_curve():
    fig, ax = plt.subplots()
    ax.plot(mlp.loss_curve_)
    ax.set_xlabel('Iterasyon')
    ax.set_ylabel('Loss Değeri')
    ax.set_title('Loss Değeri Eğrisi')
    return fig

# Sınıf bilgileri
class_counts = train_df.iloc[:, -1].value_counts()

# Öznitelik analizi (ortalama ve standart sapma)
attribute_analysis = train_df.describe().loc[['mean', 'std']]

# Öznitelik analizi metni
attribute_analysis_text = ""
for attr, values in attribute_analysis.iteritems():
    attribute_analysis_text += f"{attr}:\n  Ortalama: {values['mean']:.2f}\n  Std Sapma: {values['std']:.2f}\n\n"

window = tk.Tk()
window.title("Yapay Sinir Ağları ile Tiroid Hastalığı Tahmini")
window.geometry('1600x960')

frame = tk.Frame(window)
frame.pack(padx=10, pady=10, fill='both', expand=True)

graphics_frame = tk.LabelFrame(frame, text="Grafikler", font=('Arial', 11, 'bold'))
graphics_frame.grid(row=0, column=0, padx=10, pady=10, sticky='ns')

fig1 = plot_confusion_matrix(cm)
canvas1 = FigureCanvasTkAgg(fig1, master=graphics_frame)
canvas1.draw()
canvas1.get_tk_widget().pack(padx=5, pady=5)

fig2 = plot_loss_curve()
canvas2 = FigureCanvasTkAgg(fig2, master=graphics_frame)
canvas2.draw()
canvas2.get_tk_widget().pack(padx=5, pady=5)

report_frame = tk.LabelFrame(frame, text="Sınıflandırma Raporu ve Doğruluk Değerleri", font=('Arial', 11, 'bold'))
report_frame.grid(row=0, column=1, padx=10, pady=10, sticky='n')

report_text = display_classification_report(cr)
report_lines = report_text.split('\n')

for line in report_lines:
    label = tk.Label(report_frame, text=line, justify='left', anchor='w', font=('Arial', 11))
    label.pack(anchor='w')

def on_leave(event):
    if hasattr(event.widget, 'tooltip'):
        event.widget.tooltip.destroy()

def on_enter(event, attribute):
    text = attribute_descriptions.get(attribute, "Bilgi yok.")
    tooltip = tk.Toplevel(window, bg="white")
    tooltip.wm_overrideredirect(True)
    tooltip.geometry(f"+{event.x_root+10}+{event.y_root+10}")
    label = tk.Label(tooltip, text=text, bg="white", font=('Arial', 10, 'normal'), wraplength=300)
    label.pack()
    event.widget.tooltip = tooltip

input_frame = tk.LabelFrame(frame, text="Öznitelikler", font=('Arial', 11, 'bold'))
input_frame.grid(row=0, column=2, padx=10, pady=10, sticky='n')

attributes = [
    ("Yaş", "0.01 - 0.94"), 
    ("Cinsiyet", "0: Erkek, 1: Kadın"), 
    ("Tiroksin kullanımı", "0: Hayır, 1: Evet"), 
    ("Tiroksin sorgulaması", "0: Hayır, 1: Evet"), 
    ("Antitiroidi ilaç kullanımı", "0: Hayır, 1: Evet"), 
    ("Hasta", "0: Hayır, 1: Evet"), 
    ("Hamile", "0: Hayır, 1: Evet"), 
    ("Tiroid ameliyatı", "0: Hayır, 1: Evet"), 
    ("I131 tedavisi", "0: Hayır, 1: Evet"), 
    ("Hipotiroidi şüphesi", "0: Hayır, 1: Evet"), 
    ("Hipertiroidi şüphesi", "0: Hayır, 1: Evet"), 
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

attribute_descriptions = {
    "Yaş": "Hastanın yaşı.",
    "Cinsiyet": "Hastanın Cinsiyeti",
    "Tiroksin kullanımı": "Tiroksin kullanıyor mu",
    "Tiroksin sorgulaması": "Tiroksin tedavisinin uygulandı mı",
    "Antitiroidi ilaç kullanımı": "Tiroid'e karşı bir ilaç kullanıyor mu",
    "Hasta": "Herhangi bir hastalığı var mı.",
    "Hamile": "Hasta hamile mi",
    "Tiroid ameliyatı": "Tiroid ameliyatı oldu mu",
    "I131 tedavisi": "I131 radyoaktif tedavisi uygulanıldı mı",
    "Hipotiroidi şüphesi": "Ön değerlendirme sonucu",
    "Hipertiroidi şüphesi": "Ön değerlendirme sonucu",
    "Lityum": "Lityum tedavisi alıyor mu.",
    "Guatr": "Guatr hastalığı var mı",
    "Tümör": "Tümör hastalığı var mı",
    "Hipopitüiter": "Hipopitüiter hastalığı var mı",
    "TSH değeri": "Tiroid Uyarıcı Hormon seviyesi.",
    "T3 değeri": "Triiyodotironin seviyesi, T3 tiroid hormonu seviyesi",
    "TT4 değeri": "Toplam Tiroksin seviyesi.",
    "T4U değeri": "Tiroksin Bağlanma Kapasitesi.",
    "FTI değeri": "Serbest Tiroksin İndeksi.",
    "Yönlendirme kaynağı": "Hastanın yönlendirildiği kaynak."
}

entries = []
for i, (attribute, value_range) in enumerate(attributes):
    label = tk.Label(input_frame, text=f'{attribute} ({value_range}):', font=('Arial', 11))
    label.grid(row=i, column=0, padx=5, pady=2, sticky='e')
    entry = tk.Entry(input_frame)
    entry.grid(row=i, column=1, padx=5, pady=2)
    entries.append(entry)
    label.bind("<Enter>", lambda e, attribute=attribute: on_enter(e, attribute))
    label.bind("<Leave>", on_leave)

predict_button = tk.Button(input_frame, text="Tahmin Et", command=predict, font=('Arial', 11, 'bold'), bg='#84d4f5')
predict_button.grid(row=len(attributes), column=0, columnspan=2, pady=10)

example_frame = tk.LabelFrame(input_frame, text="Örnek Veriler", font=('Arial', 11, 'bold'))
example_frame.grid(row=len(attributes)+1, column=0, columnspan=2, pady=10)

example_normal_button1 = tk.Button(example_frame, text="Normal Örnek 1", command=lambda: fill_example(example_normal[0]), font=('Arial', 11), bg='#b8ffb8')
example_normal_button1.grid(row=0, column=0, padx=5, pady=2)

example_normal_button2 = tk.Button(example_frame, text="Normal Örnek 2", command=lambda: fill_example(example_normal[1]), font=('Arial', 11), bg='#b8ffb8')
example_normal_button2.grid(row=0, column=1, padx=5, pady=2)

example_hipertiroidi_button1 = tk.Button(example_frame, text="Hipertiroidi Örnek 1", command=lambda: fill_example(example_hipertiroidi[0]), font=('Arial', 11), bg='#ffb89c')
example_hipertiroidi_button1.grid(row=1, column=0, padx=5, pady=2)

example_hipertiroidi_button2 = tk.Button(example_frame, text="Hipertiroidi Örnek 2", command=lambda: fill_example(example_hipertiroidi[1]), font=('Arial', 11), bg='#ffb89c')
example_hipertiroidi_button2.grid(row=1, column=1, padx=5, pady=2)

example_hipotiroidi_button1 = tk.Button(example_frame, text="Hipotiroidi Örnek 1", command=lambda: fill_example(example_hipotiroidi[0]), font=('Arial', 11), bg='#f57171')
example_hipotiroidi_button1.grid(row=2, column=0, padx=5, pady=2)

example_hipotiroidi_button2 = tk.Button(example_frame, text="Hipotiroidi Örnek 2", command=lambda: fill_example(example_hipotiroidi[1]), font=('Arial', 11), bg='#f57171')
example_hipotiroidi_button2.grid(row=2, column=1, padx=5, pady=2)

result_label = tk.Label(input_frame, text="Tahmin: ", font=('Arial', 13, 'bold'), fg='black')
result_label.grid(row=len(attributes)+2, column=0, columnspan=2, pady=10)

# Veri Seti Bilgileri Çerçevesi
dataset_info_frame = tk.LabelFrame(frame, text="Veri Seti Bilgileri", font=('Arial', 11, 'bold'))
dataset_info_frame.grid(row=0, column=3, padx=10, pady=10, sticky='n')

# Sınıf Bilgileri
class_info_label = tk.Label(dataset_info_frame, text="Sınıf Bilgileri:", font=('Arial', 11, 'bold'))
class_info_label.pack(anchor='w', padx=5, pady=2)

for cls, count in class_counts.items():
    cls_label = tk.Label(dataset_info_frame, text=f"Sınıf {int(cls)}: {count} örnek", font=('Arial', 11))
    cls_label.pack(anchor='w', padx=5, pady=2)

# Öznitelik Analizi Bilgileri
attr_info_label = tk.Label(dataset_info_frame, text="Öznitelik Analizi:", font=('Arial', 11, 'bold'))
attr_info_label.pack(anchor='w', padx=5, pady=2)

attr_info_text = tk.Text(dataset_info_frame, wrap='word', height=15, font=('Arial', 11))
attr_info_text.pack(fill='both', expand=True, padx=5, pady=2)
attr_info_text.insert('1.0', attribute_analysis_text)
attr_info_text.config(state=tk.DISABLED)

window.mainloop()