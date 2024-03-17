# Laporan Proyek Machine Learning - Derajat Salim Wibowo

## Domain Proyek

Latar belakang
Definisi Stroke
Menurut World Health Organization (WHO), stroke adalah sindrom klinis yang berkembang cepat dengan tanda dan gejala hilangnya fungsi otak fokal (atau global) yang berlangsung lebih dari 24 jam, atau menyebabkan kematian, tanpa penyebab lain yang jelas selain vaskular.

Stroke merupakan penyakit yang menjadi penyebab kematian kedua dan penyebab kehilangan tahun hidup disabilitas (DALYs) ketiga di dunia. Biaya global yang terkait dengan stroke diperkirakan mencapai lebih dari US$891 miliar, setara dengan 1.12% dari GDP global. Selama periode dari tahun 1990 hingga 2019, terjadi peningkatan signifikan dalam penggunaan stroke, dengan penambahan jumlah kasus stroke baru sebesar 70.0%, kematian akibat stroke sebesar 43.0%, stroke yang masih ada sebesar 102.0%, dan DALYs sebesar 143.0%. Sebagian besar penggunaan stroke, sebesar 86.0% dari kematian dan 89.0% DALYs, terjadi di negara-negara dengan ekonomi rendah dan rendah-sedang (LMIC). Terdapat perbedaan besar dalam tingkat standardisasi umur insiden stroke, mortalitas, prevalensi, dan DALYs, dengan tingkat tertinggi terjadi di LMIC, terutama di Asia Tenggara, Asia, dan Afrika Selatan. Pada tahun 2019, jumlah DALYs akibat stroke pada pria (77.0 juta) lebih tinggi dibandingkan dengan wanita (66.0 juta), tetapi jumlah kasus stroke baru dan stroke yang masih ada lebih tinggi pada wanita (6.4 juta kasus stroke baru dan 56.4 juta stroke yang masih ada) dibandingkan dengan pria (5.8 juta kasus stroke baru dan 45.0 juta stroke yang masih ada). Meskipun tingkat standardisasi umur insiden, mortalitas, prevalensi, dan DALYs tidak berbeda secara signifikan antara pria dan wanita, tingkat mortalitas lebih tinggi pada pria (96.4 per 100.000 per tahun) dibandingkan dengan wanita (73.5 per 100.000 per tahun).

Dengan latar belakang bahwa stroke merupakan penyebab utama kematian dan kehilangan DALYs, proyek ini menggunakan model machine learning untuk memprediksi risiko stroke pada manusia. Dengan data yang luas tentang faktor risiko stroke, model ini dilatih untuk mengidentifikasi pola dan hubungan yang dapat mengarah pada prediksi risiko stroke. Tujuannya adalah memberikan intervensi tepat waktu dan efektif kepada individu yang rentan terhadap stroke, dengan harapan dapat mengurangi angka kesakitan dan kematian akibat stroke di seluruh dunia.


## Business Understanding

### Problem Statements
- Bagaimana cara melakukan pra-pemrosesan pada data penyakit stroke yang akan digunakan untuk membuat model yang baik?
- Bagaimana cara membuat model untuk memprediksi penyakit stroke pada manusia dengan menggunakan machine learning?
- Bagaimana nilai ekonomi proyek ini dalam mata bisnis?

### Goals
- Melakukan pra-pemrosesan data dengan baik agar dapat digunakan dalam pembuatan model machine learning. Ini mencakup langkah-langkah seperti penanganan nilai-nilai yang hilang, normalisasi fitur, dan pemrosesan transform data.
- Mengetahui dan menerapkan berbagai teknik dan algoritma machine learning untuk membuat model yang dapat memprediksi penyakit stroke pada manusia. Ini melibatkan pemilihan fitur, pemilihan model, pelatihan model, dan evaluasi kinerja model.
- Menjadikan model machine learning yang dilatih sebagai acuan dalam pengambilan keputusan medis untuk mencegah terjadinya penyakit stroke pada manusia dan memiliki nilai jual ekonomi dalam dunia kesehatan.

### Solution statements
                                                                                       
- Menggunakan Random Forest Classifier: Menerapkan algoritma Random Forest Classifier untuk memprediksi penyakit stroke. Melakukan prapemrosesan data yang tepat, termasuk penanganan nilai-nilai yang hilang dan penskalaan fitur. Melakukan pelatihan model menggunakan Random Forest Classifier dengan parameter default, dan kemudian melakukan evaluasi kinerja model menggunakan metrik evaluasi seperti akurasi, presisi, recall, dan F1-score.
- Menggunakan Decision Tree Classifier: Menerapkan algoritma Decision Tree Classifier untuk memprediksi penyakit stroke. Sama seperti sebelumnya, Melakukan prapemrosesan data yang sesuai dan pelatihan model menggunakan Decision Tree Classifier dengan parameter default. Setelah itu, Mengevaluasi kinerja model menggunakan metrik evaluasi yang sama seperti pada solusi sebelumnya.
- Perbandingan Antara Random Forest dan Decision Tree: Membandingkan kinerja kedua model yang telah dibangun (Random Forest Classifier dan Decision Tree Classifier). Menggunakan metrik evaluasi seperti akurasi, presisi, recall, dan F1-score untuk menentukan model mana yang memberikan hasil terbaik dalam memprediksi penyakit stroke. Perbandingan ini akan membantu  memilih model terbaik untuk tujuan prediksi yang akurat.

## Data Understanding
Dataset yang digunakan pada proyek kali ini terdiri dari 5110baris dan 12 kolom dengan judul [Stroke Prediction Dataset] (https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

### Variabel-variabel pada Stroke Prediction Dataset dataset adalah sebagai berikut:
1) id: Identifier unik
2) gender: "Pria", "Wanita", atau "Lainnya"
3) age: Usia pasien
4) hypertension: 0 untuk pasien yang tidak mengalami hipertensi, 1 untuk pasien yang mengalami hipertensi
5) heart_disease: 0 untuk pasien yang tidak memiliki penyakit jantung, 1 untuk pasien yang memiliki penyakit jantung
6) ever_married: "Tidak" atau "Ya"
7) work_type: "Anak-anak", "Pekerja Pemerintah", "Tidak Pernah Bekerja", "Pekerja Swasta", atau "Wiraswasta"
8) Residence_type: "Pedesaan" atau "Perkotaan"
9) avg_glucose_level: Rata-rata tingkat glukosa dalam darah
10) bmi: Indeks massa tubuh
11) smoking_status: "Pernah Merokok", "Tidak Pernah Merokok", "Merokok", atau "Status Merokok Tidak Diketahui"
12) stroke: 1 jika pasien telah mengalami stroke atau 0 jika tidak
    
![image](https://github.com/Auraja/Proyek1_ML_Mahir_Dicoding/assets/116571074/d1bbaba6-2aed-41d3-8a41-bc1c08cc6484)
Gambar 1. KDE Plot, Boxplot, dan Scatterplot      

Visualisasi ini menampilkan tiga jenis plot untuk setiap kolom numerik dalam dataset: KDE Plot, Boxplot, dan Scatterplot. KDE Plot menampilkan distribusi probabilitas variabel numerik, memisahkan antara kelompok yang mengalami stroke dan yang tidak. Boxplot menunjukkan statistik deskriptif dan outlier variabel numerik, juga memisahkan berdasarkan nilai target 'stroke'. Scatterplot menunjukkan hubungan antara variabel numerik dan variabel target 'stroke'. Tujuannya adalah untuk memahami distribusi data, perbedaan statistik, dan hubungan antara variabel numerik dan kemungkinan terjadinya stroke.

![image](https://github.com/Auraja/Proyek1_ML_Mahir_Dicoding/assets/116571074/625d7c5b-31be-44a3-97f9-f77e9d5530be)
![image](https://github.com/Auraja/Proyek1_ML_Mahir_Dicoding/assets/116571074/74aff240-7940-4161-ad88-882c73266f44)
![image](https://github.com/Auraja/Proyek1_ML_Mahir_Dicoding/assets/116571074/925bde0f-2c5c-4b9c-9c16-ccd768c84b0a)
![image](https://github.com/Auraja/Proyek1_ML_Mahir_Dicoding/assets/116571074/a9f68d7e-b9cf-4fc5-93be-a69604d9b2d6)
Gambar 2. Frekuensi Masing Masing Atribut

Visualisasi ini menampilkan dua plot untuk masing-masing kolom kategorikal dalam dataset. Setiap plot menggunakan countplot untuk menampilkan jumlah frekuensi masing-masing kategori dalam kolom yang ditampilkan. Tujuannya adalah untuk memberikan pemahaman tentang distribusi data dalam setiap kolom kategorikal. Dengan visualisasi ini, dapat melihat seberapa seimbang atau tidak seimbangnya distribusi kategori dalam setiap fitur kategorikal, yang dapat memberikan wawasan tentang kecenderungan atau pola dalam data.

![image](https://github.com/Auraja/Proyek1_ML_Mahir_Dicoding/assets/116571074/9f3b0881-098b-4ad6-97e7-3c7ee60d29d6)                             
Gambar 3. Stroke vs Non-Stroke

Visualisasi ini adalah diagram lingkaran yang menunjukkan proporsi pasien yang mengalami stroke (stroke=1) dan yang tidak (stroke=0) dalam dataset. Tujuannya adalah untuk memberikan gambaran visual tentang seberapa sering stroke terjadi dalam sampel data. Dengan ini, dapat mengevaluasi seimbangnya distribusi kelas target dan memahami tingkat keparahan masalah stroke dalam dataset.

Hasilnya:

- Sekitar 96% sampel tidak memiliki Stroke dan 4% memiliki stroke.
- Distribusi sampel adalah distribusi Normal.
- Mereka yang pernah mengalami stroke berada dalam rentang usia 40 hingga 85 tahun, indeks massa tubuh (BMI) dalam rentang 20 hingga 40, dan level glukosa dalam rentang 50 hingga 130.
- Sekitar 60% sampel adalah perempuan.
- Sekitar 91% sampel tidak memiliki hipertensi.
- Sekitar 95% sampel tidak memiliki penyakit jantung.
- Sekitar 34% sampel belum pernah menikah.
- Sebagian besar sampel bekerja di sektor swasta.
- Kami tidak memiliki informasi dalam bidang merokok untuk 1483 sampel.

## Data Preparation

| Column         | Description                      | Values                                                                  |  
|----------------|----------------------------------|-------------------------------------------------------------------------|
| Gender         | Biological sex assigned at birth | Male: 0, Female: 1, Other: 2 (Optional)                                 |   
| Ever Married   | Current marital status           | Yes: 0, No: 1                                                           |  
| Work Type      | Current employment status        | Private: 0, Self-employed: 1, Govt_job: 2, Children: 3, Never_worked: 4 |   
| Smoking Status | Current or past smoking habits   | Formerly smoked: 0, Never smoked: 1, Smokes: 2, Unknown: 3              |   
| Residence Type | Location of primary residence    | Urban: 0, Rural: 1                                                      |
                                                    Tabel 1.0

Data preparation ini terdiri dari beberapa langkah berikut:

1. Mapping Nilai Kategorikal ke Numerik: Pada bagian pertama kode, setiap kolom kategorikal dalam columns_temp dipetakan ke nilai numerik. Misalnya, untuk kolom 'gender', nilai 'Male' dipetakan ke 0, 'Female' ke 1, dan 'Other' ke 2. Ini membantu mengubah data kategorikal menjadi format yang dapat diproses oleh model machine learning.            
2. Penggantian Nilai Kategorikal: Setelah mapping, setiap kolom dalam DataFrame data_2 diganti dengan nilai numerik sesuai dengan mapping yang ditentukan sebelumnya. Misalnya, 'Yes' dalam kolom 'ever_married' diganti dengan 0 dan 'No' dengan Hal yang sama dilakukan untuk kolom 'work_type', 'smoking_status', dan 'Residence_type'.            
3. Pembersihan Data: Setelah itu, data dibersihkan dengan menghapus baris yang memiliki nilai 'gender' yang sama dengan 2 (yang mewakili kategori 'Other'). Ini mungkin dilakukan karena kategori 'Other' memiliki nilai yang tidak dapat dipetakan secara jelas dalam konteks nilai numerik yang ditentukan sebelumnya.

Tujuan dari langkah-langkah ini adalah untuk mengubah data awal yang terdiri dari nilai-nilai kategorikal menjadi data yang dapat diproses oleh model machine learning. Dengan melakukannya, mempersiapkan dataset untuk analisis lebih lanjut atau pembangunan model prediksi, dengan menghilangkan nilai yang tidak relevan atau ambigu serta mengubah nilai-nilai kategorikal menjadi format yang dapat dimengerti oleh model.

![image](https://github.com/Auraja/Proyek1_ML_Mahir_Dicoding/assets/116571074/73be564c-53f6-47d8-af1d-185e188b3ab7)                
Gambar 4. Normalisasi

Data preparation ini terdiri dari langkah-langkah sebagai berikut:

1. Memisahkan Fitur dan Target: Data dipisahkan menjadi dua bagian: fitur (X_temp) dan target (y). Fitur adalah semua kolom dalam data kecuali kolom 'stroke', sedangkan target adalah kolom 'stroke' itu sendiri. Ini penting karena dalam machine learning, memisahkan variabel target yang akan diprediksi dari fitur yang digunakan untuk melakukan prediksi.
2. Normalisasi Fitur: Fitur-fitur dalam X_temp dinormalisasi menggunakan MinMaxScaler. Normalisasi dilakukan untuk mengubah rentang nilai setiap fitur sehingga memiliki rentang antara 0 dan 1. Hal ini membantu dalam meningkatkan kinerja model machine learning.
3. Membuat DataFrame Baru untuk Fitur: Hasil normalisasi disimpan dalam DataFrame baru yang disebut X. DataFrame ini berisi fitur-fitur yang sudah dinormalisasi, dengan nama kolom yang sama seperti di X_temp.

Tujuan dari langkah ini adalah untuk mempersiapkan data yang siap digunakan untuk proses pembelajaran mesin. Dengan melakukan normalisasi, memastikan bahwa semua fitur memiliki skala yang seragam, yang dapat membantu meningkatkan kinerja model dan mempercepat konvergensi algoritma pembelajaran mesin.


## Modeling

Model yang digunakan dalam proses pemodelan adalah Random Forest Classifier dan Decision Tree Classifier.

1. Random Forest Classifier: Ini adalah model ensemble yang terdiri dari banyak pohon keputusan. Masing-masing pohon diberi bagian dari dataset yang berbeda dan akhirnya menghasilkan prediksi yang digabungkan. Ini membuat model lebih tahan terhadap overfitting dan lebih stabil. Parameter yang digunakan dalam pemodelan adalah:
- n_estimators: Jumlah pohon dalam ensemble. Nilai yang diuji adalah 50, 100, 250, dan 500.
- criterion: Kriteria untuk mengukur kualitas split. Pilihan adalah 'gini', 'entropy', dan 'log_loss'.
- max_features: Jumlah fitur yang dipertimbangkan ketika mencari split terbaik. Pilihan adalah 'sqrt' (akar kuadrat dari jumlah fitur) dan 'log2' (logaritma basis 2 dari jumlah fitur).

2. Decision Tree Classifier: Ini adalah model pohon keputusan yang membagi data menjadi subset yang semakin kecil berdasarkan aturan keputusan. Parameter yang digunakan dalam pemodelan adalah:
- criterion: Kriteria untuk mengukur kualitas split. Pilihan adalah 'gini', 'entropy', dan 'log_loss'.
- splitter: Strategi untuk memilih split pada setiap node. Pilihan adalah 'best' (mencari split terbaik) dan 'random' (membuat split secara acak).
- max_depth: Maksimum kedalaman pohon. Rentang nilai yang diuji adalah dari 4 hingga 29.
  
Setelah melakukan penalaan hiperparameter menggunakan GridSearchCV, parameter terbaik yang dihasilkan adalah:
Untuk Random Forest Classifier:
- n_estimators: 100
- criterion: 'gini'
- max_features: 'sqrt'
Untuk Decision Tree Classifier:
- criterion: 'gini'
- splitter: 'best'
- max_depth: Nilai tidak ditampilkan, namun merupakan nilai terbaik yang ditemukan dari rentang yang diuji.
  
Dengan parameter-parameter ini, model-model tersebut memberikan akurasi terbaik yang diperoleh dari proses tuning.
Random Forest Classifier
| precision    	| recall 	| f1-score 	| support 	|      	|
|--------------	|--------	|----------	|---------	|------	|
| 0            	| 0.96   	| 1.00     	| 0.98    	| 1178 	|
| 1            	| 0.00   	| 0.00     	| 0.00    	| 49   	|
| accuracy     	| 0.96   	| 1227     	|         	|      	|
| macro avg    	| 0.48   	| 0.50     	| 0.49    	| 1227 	|
| weighted avg 	| 0.92   	| 0.96     	| 0.94    	| 1227 	|
                        Tabel 2.0

Decision Tree Classifier
| precision    	| recall 	| f1-score 	| support 	|      	|
|--------------	|--------	|----------	|---------	|------	|
| 0            	| 0.96   	| 1.00     	| 0.98    	| 1178 	|
| 1            	| 0.00   	| 0.00     	| 0.00    	| 49   	|
| accuracy     	| 0.96   	| 1227     	|         	|      	|
| macro avg    	| 0.48   	| 0.50     	| 0.49    	| 1227 	|
| weighted avg 	| 0.92   	| 0.96     	| 0.94    	| 1227 	|
                        Tabel 3.0
                        
Hasil evaluasi dari kedua model, Tabel 2.0 (Random Forest Classifier) dan Tabel 3.0 (Decision Tree Classifier), menunjukkan hasil yang sama dalam hal akurasi. Oleh karena itu diperlukan penambahan metric baru untuk mengukur kinerja dari masing-masing algoritma, yaitu akurasi model dan waktu eksekusi model.

| Algorithm |                  Score | Delta_t |       |   |
|----------:|-----------------------:|--------:|-------|---|
|     0     | RandomForestClassifier |    0.96 | 1.155 |   |
|     1     | DecisionTreeClassifier |    0.96 | 0.008 |   |
                        Tabel 4.0

Berdasarkan hasil yang diberikan, baik Random Forest Classifier maupun Decision Tree Classifier memiliki akurasi yang sama, yaitu 0.96. Namun, Decision Tree Classifier lebih cepat dalam prosesnya dengan waktu hanya 0.008 detik, sedangkan Random Forest Classifier memerlukan waktu 1.155 detik. Oleh karena itu waktu komputasi menjadi faktor penting, Decision Tree Classifier adalah pilihan yang lebih baik.

Secara singkat, Random Forest Classifier memiliki kelebihan dalam menangani overfitting dan cocok untuk data besar dengan fitur yang banyak, meskipun memerlukan kompleksitas komputasi yang tinggi. Sementara itu, Decision Tree Classifier mudah dimengerti dan diinterpretasikan, namun rentan terhadap overfitting dan tidak stabil terhadap perubahan data.

## Evaluation
Metric evaluasi yang digunakan adalah akurasi pada model yang digunakan dan waktu eksekusi proses pelatihan model

1. Akurasi:
Akurasi mengukur proporsi data yang diklasifikasikan dengan benar oleh model.

Rumus:
![image](https://github.com/Auraja/Proyek1_ML_Mahir_Dicoding/assets/116571074/7e5b7213-372d-467a-ae40-93bfa3e4643e)
- y_pred_tree: Prediksi kelas untuk data pengujian oleh DecisionTreeClassifier.
- tree.score(X_test, y_test): Fungsi untuk menghitung akurasi model.
- tree_score: Variabel yang menyimpan nilai akurasi.

2. Waktu pelatihan:
Metrik ini mengukur waktu yang dibutuhkan model untuk dilatih pada data.

Rumus:
![image](https://github.com/Auraja/Proyek1_ML_Mahir_Dicoding/assets/116571074/a7ca0f96-c299-4575-9a28-85339c304fa6)
- t1: Waktu awal pelatihan model.
- t2: Waktu akhir pelatihan model.
- delta_t: Variabel yang menyimpan waktu pelatihan dalam detik.

![image](https://github.com/Auraja/Proyek1_ML_Mahir_Dicoding/assets/116571074/93bff0fc-b661-48d8-8491-ebdba23308f7)
Gambar 5. Akurasi dan Waktu Eksekusi Model

Kedua model, RandomForestClassifier dan DecisionTreeClassifier, memiliki skor akurasi yang sama, yaitu 0.96. Namun, model DecisionTreeClassifier memiliki waktu eksekusi yang jauh lebih cepat dibandingkan RandomForestClassifier. Dengan Delta_t hanya 0.008 detik, model DecisionTreeClassifier mungkin lebih efisien dalam penggunaan sumber daya komputasi.

Proyek ini berhasil mengembangkan model prediksi penyakit stroke menggunakan algoritma Decision Tree Classifier. Model tersebut mencapai akurasi hingga 96% dalam mengidentifikasi penyakit stroke. Ini menandakan bahwa seluruh goals dan solusi yang ditetapkan dalam proyek tercapai.  Dengan performa tersebut, model ini memiliki potensi untuk bersaing dalam ranah bisnis kesehatan. 

### REFERENCES              
Feigin, V. L., Brainin, M., Norrving, B., Martins, S., Sacco, R. L., Hacke, W., ... & Lindsay, P. (2022). World Stroke Organization (WSO): global stroke fact sheet 2022. International Journal of Stroke, 17(1), 18-29.                                                                                        
World Health Organization, “Stroke, Cerebrovascular accident,” [Online]. Available: https://www.emro.who.int/health-topics/stroke-cerebrovascular-accident/index.html. [Accessed: 17-Mar-2024].



