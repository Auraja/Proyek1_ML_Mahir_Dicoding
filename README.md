# Laporan Proyek Machine Learning - Derajat Salim Wibowo

## Domain Proyek
Domain yang dipilih pada proyek ini adalah kesehatan yang berupa prediksi penyakit stroke pada manusia

Latar belakang
Definisi Stroke

Menurut World Health Organization (WHO), stroke adalah sindrom klinis yang berkembang cepat dengan tanda dan gejala hilangnya fungsi otak fokal (atau global) yang berlangsung lebih dari 24 jam, atau menyebabkan kematian, tanpa penyebab lain yang jelas selain vaskular.

Statistik Global

Stroke merupakan penyebab kematian nomor dua di dunia, setelah penyakit jantung koroner, dan penyebab kecacatan utama pada orang dewasa.
Pada tahun 2020, diperkirakan 13,7 juta orang mengalami stroke di seluruh dunia, dan 5,5 juta meninggal karenanya.
Dari tahun 2000 hingga 2020, angka kematian akibat stroke telah menurun 25%, namun stroke masih menjadi masalah kesehatan utama di seluruh dunia.
Faktor Risiko Stroke

Faktor risiko utama stroke adalah:

Hipertensi
Merokok
Dislipidemia
Diabetes mellitus
Fibrilasi atrium
Penyalahgunaan alkohol
Diet yang tidak sehat
Kurang aktivitas fisik

Mengapa Stroke Harus Diselesaikan?

Stroke merupakan masalah kesehatan utama di dunia dengan dampak signifikan pada kesehatan individu, keluarga, dan masyarakat. Berikut beberapa alasan mengapa stroke harus diatasi:

1. Beban Penyakit Tinggi:
Stroke adalah penyebab kematian nomor dua di dunia, setelah penyakit jantung koroner.
Pada tahun 2020, stroke menyebabkan 5,5 juta kematian di seluruh dunia.
Di Indonesia, stroke merupakan penyebab kematian nomor satu.
Stroke juga menyebabkan kecacatan permanen pada banyak orang.

2. Dampak Ekonomi:
Biaya perawatan stroke sangat tinggi.
Stroke dapat menyebabkan hilangnya produktivitas dan pendapatan.
Beban ekonomi stroke dapat membebani keluarga dan masyarakat.

3. Kualitas Hidup:
Stroke dapat menyebabkan berbagai masalah fisik, mental, dan emosional.
Kualitas hidup penyintas stroke dan keluarganya dapat menurun drastis.

Bagaimana Mengatasi Stroke?

1. Pencegahan:
Pencegahan stroke adalah kunci utama untuk mengatasi masalah ini.
Faktor risiko stroke seperti hipertensi, diabetes, dan merokok harus dikendalikan.
Gaya hidup sehat seperti diet seimbang, olahraga teratur, dan berhenti merokok harus dipromosikan.

2. Pengobatan:
Pengobatan stroke yang tepat dan cepat dapat meningkatkan peluang pemulihan.
Obat-obatan dan terapi rehabilitasi dapat membantu penyintas stroke untuk kembali ke kehidupan normal.

3. Peningkatan Kesadaran:
Kesadaran masyarakat tentang stroke harus ditingkatkan.
Masyarakat harus di edukasi tentang tanda-tanda dan gejala stroke, serta cara pencegahan dan pengobatannya.

[Kementerian Kesehatan Republik Indonesia. (2023). Stroke.] (https://p2ptm.kemkes.go.id/infographic-p2ptm/stroke/apa-itu-stroke)
[Ikatan Dokter Indonesia. (2023). Stroke.] (https://www.ahajournals.org/doi/abs/10.1161/STR.0000000000000430)

## Business Understanding

Pada bagian ini, kamu perlu menjelaskan proses klarifikasi masalah.

Bagian laporan ini mencakup:

### Problem Statements
- Bagaimana cara melakukan pra-pemrosesan pada data penyakit stroke yang akan digunakan untuk membuat model yang baik?
- Bagaimana cara membuat model untuk memprediksi penyakit stroke pada manusia dengan menggunakan machine learning?

### Goals
- Melakukan pra-pemrosesan data dengan baik agar dapat digunakan dalam pembuatan model machine learning. Ini mencakup langkah-langkah seperti penanganan nilai-nilai yang hilang, normalisasi fitur, dan pemrosesan transform data.
- Mengetahui dan menerapkan berbagai teknik dan algoritma machine learning untuk membuat model yang dapat memprediksi penyakit stroke pada manusia. Ini melibatkan pemilihan fitur, pemilihan model, pelatihan model, dan evaluasi kinerja model.

Semua poin di atas harus diuraikan dengan jelas. Anda bebas menuliskan berapa pernyataan masalah dan juga goals yang diinginkan.

### Solution statements
Solution Statements
-Menggunakan Random Forest Classifier: Kami akan menerapkan algoritma Random Forest Classifier untuk memprediksi penyakit stroke. Kami akan melakukan prapemrosesan data yang tepat, termasuk penanganan nilai-nilai yang hilang dan penskalaan fitur. Kami akan melakukan pelatihan model menggunakan Random Forest Classifier dengan parameter default, dan kemudian melakukan evaluasi kinerja model menggunakan metrik evaluasi seperti akurasi, presisi, recall, dan F1-score.
-Menggunakan Decision Tree Classifier: Kami akan menerapkan algoritma Decision Tree Classifier untuk memprediksi penyakit stroke. Sama seperti sebelumnya, kami akan melakukan prapemrosesan data yang sesuai dan pelatihan model menggunakan Decision Tree Classifier dengan parameter default. Setelah itu, kami akan mengevaluasi kinerja model menggunakan metrik evaluasi yang sama seperti pada solusi sebelumnya.
-Perbandingan Antara Random Forest dan Decision Tree: Kami akan membandingkan kinerja kedua model yang telah dibangun (Random Forest Classifier dan Decision Tree Classifier). Kami akan menggunakan metrik evaluasi seperti akurasi, presisi, recall, dan F1-score untuk menentukan model mana yang memberikan hasil terbaik dalam memprediksi penyakit stroke. Perbandingan ini akan membantu kami memilih model terbaik untuk tujuan prediksi yang akurat.

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
    
![image](https://github.com/Auraja/test-md/assets/116571074/6ac5c42d-cec5-4c55-b7f0-fb9392d53aa0)
Visualisasi ini menampilkan tiga jenis plot untuk setiap kolom numerik dalam dataset: KDE Plot, Boxplot, dan Scatterplot. KDE Plot menampilkan distribusi probabilitas variabel numerik, memisahkan antara kelompok yang mengalami stroke dan yang tidak. Boxplot menunjukkan statistik deskriptif dan outlier variabel numerik, juga memisahkan berdasarkan nilai target 'stroke'. Scatterplot menunjukkan hubungan antara variabel numerik dan variabel target 'stroke'. Tujuannya adalah untuk memahami distribusi data, perbedaan statistik, dan hubungan antara variabel numerik dan kemungkinan terjadinya stroke.

![image](https://github.com/Auraja/test-md/assets/116571074/2e02cb9c-dc60-4f37-8ea6-df68bef10d34)
![image](https://github.com/Auraja/test-md/assets/116571074/ecd76e21-d138-4021-8dc5-539e30a384e7)
![image](https://github.com/Auraja/test-md/assets/116571074/67390e5e-2b6a-46d8-aeda-0438b9707fca)
![image](https://github.com/Auraja/test-md/assets/116571074/3baacfaf-db17-4863-8465-570a919ad6cd)
Visualisasi ini menampilkan dua plot untuk masing-masing kolom kategorikal dalam dataset. Setiap plot menggunakan countplot untuk menampilkan jumlah frekuensi masing-masing kategori dalam kolom yang ditampilkan. Tujuannya adalah untuk memberikan pemahaman tentang distribusi data dalam setiap kolom kategorikal. Dengan visualisasi ini, kita dapat melihat seberapa seimbang atau tidak seimbangnya distribusi kategori dalam setiap fitur kategorikal, yang dapat memberikan wawasan tentang kecenderungan atau pola dalam data.

![image](https://github.com/Auraja/test-md/assets/116571074/60c32213-a388-47ff-b1a4-6ade8755085d)
Visualisasi ini adalah diagram lingkaran yang menunjukkan proporsi pasien yang mengalami stroke (stroke=1) dan yang tidak (stroke=0) dalam dataset. Tujuannya adalah untuk memberikan gambaran visual tentang seberapa sering stroke terjadi dalam sampel data. Dengan ini, kita dapat mengevaluasi seimbangnya distribusi kelas target dan memahami tingkat keparahan masalah stroke dalam dataset.

## Data Preparation
![image](https://github.com/Auraja/test-md/assets/116571074/7dc10464-6958-4a5a-b155-35e0c187a20a)
![image](https://github.com/Auraja/test-md/assets/116571074/6391978e-dc03-4580-a6ec-a66958214114)
Data preparation ini terdiri dari beberapa langkah berikut:

1. Mapping Nilai Kategorikal ke Numerik: Pada bagian pertama kode, setiap kolom kategorikal dalam columns_temp dipetakan ke nilai numerik. Misalnya, untuk kolom 'gender', nilai 'Male' dipetakan ke 0, 'Female' ke 1, dan 'Other' ke 2. Ini membantu mengubah data kategorikal menjadi format yang dapat diproses oleh model machine learning.
2. Penggantian Nilai Kategorikal: Setelah mapping, setiap kolom dalam DataFrame data_2 diganti dengan nilai numerik sesuai dengan mapping yang ditentukan sebelumnya. Misalnya, 'Yes' dalam kolom 'ever_married' diganti dengan 0 dan 'No' dengan Hal yang sama dilakukan untuk kolom 'work_type', 'smoking_status', dan 'Residence_type'.
3. Pembersihan Data: Setelah itu, data dibersihkan dengan menghapus baris yang memiliki nilai 'gender' yang sama dengan 2 (yang mewakili kategori 'Other'). Ini mungkin dilakukan karena kategori 'Other' memiliki nilai yang tidak dapat dipetakan secara jelas dalam konteks nilai numerik yang ditentukan sebelumnya.

Tujuan dari langkah-langkah ini adalah untuk mengubah data awal yang terdiri dari nilai-nilai kategorikal menjadi data yang dapat diproses oleh model machine learning. Dengan melakukannya, kita mempersiapkan dataset untuk analisis lebih lanjut atau pembangunan model prediksi, dengan menghilangkan nilai yang tidak relevan atau ambigu serta mengubah nilai-nilai kategorikal menjadi format yang dapat dimengerti oleh model.

![image](https://github.com/Auraja/test-md/assets/116571074/4bd59409-e80e-482f-bf0e-4fa5e76d0da4)
Data preparation ini terdiri dari langkah-langkah sebagai berikut:

1. Memisahkan Fitur dan Target: Data dipisahkan menjadi dua bagian: fitur (X_temp) dan target (y). Fitur adalah semua kolom dalam data kecuali kolom 'stroke', sedangkan target adalah kolom 'stroke' itu sendiri. Ini penting karena dalam machine learning, kita memisahkan variabel target yang akan diprediksi dari fitur yang digunakan untuk melakukan prediksi.
2. Normalisasi Fitur: Fitur-fitur dalam X_temp dinormalisasi menggunakan MinMaxScaler. Normalisasi dilakukan untuk mengubah rentang nilai setiap fitur sehingga memiliki rentang antara 0 dan 1. Hal ini membantu dalam meningkatkan kinerja model machine learning.
3. Membuat DataFrame Baru untuk Fitur: Hasil normalisasi disimpan dalam DataFrame baru yang disebut X. DataFrame ini berisi fitur-fitur yang sudah dinormalisasi, dengan nama kolom yang sama seperti di X_temp.

Tujuan dari langkah ini adalah untuk mempersiapkan data yang siap digunakan untuk proses pembelajaran mesin. Dengan melakukan normalisasi, kita memastikan bahwa semua fitur memiliki skala yang seragam, yang dapat membantu meningkatkan kinerja model dan mempercepat konvergensi algoritma pembelajaran mesin.


## Modeling

Model pertama menggunakan algoritma Random Forest Classifier dengan GridSearchCV untuk melakukan penyetelan hiperparameter. GridSearchCV digunakan untuk mencari kombinasi terbaik dari parameter-parameter yang telah ditentukan sebelumnya. Setelah mendapatkan parameter terbaik, model Random Forest dilatih ulang menggunakan parameter tersebut. Setelah pelatihan, model dievaluasi menggunakan data uji, dan akurasinya dicetak. Selain itu, waktu eksekusi model juga diukur sebelum dan sesudah pelatihan untuk memperoleh estimasi durasi pelatihan.
Kelebihan dari Random Forest Classifier antara lain kinerja yang tinggi, kemampuan menangani overfitting, dan kemampuan menangani data yang besar serta fitur kategorikal. Namun, kekurangannya termasuk komputasi yang intensif dan kurangnya interpretasi yang mudah dipahami dari model tersebut.

Model kedua menggunakan algoritma Decision Tree Classifier dengan GridSearchCV untuk melakukan penyetelan hiperparameter. GridSearchCV digunakan untuk mencari kombinasi terbaik dari parameter-parameter yang telah ditentukan sebelumnya. Setelah mendapatkan parameter terbaik, model Decision Tree dilatih ulang menggunakan parameter tersebut. Setelah pelatihan, model dievaluasi menggunakan data uji, dan akurasinya dicetak. Selain itu, waktu eksekusi model juga diukur sebelum dan sesudah pelatihan untuk memperoleh estimasi durasi pelatihan.
Kelebihan dari Decision Tree Classifier antara lain interpretasi yang mudah, tidak memerlukan normalisasi, dan kemampuan menangani fitur kategorikal. Namun, kekurangannya termasuk rentan terhadap overfitting dan kehilangan informasi pada kedalaman pohon yang besar.

![image](https://github.com/Auraja/test-md/assets/116571074/b494ecf3-aeed-44b2-92f9-df0cfcd0c30f)
Random Forest Classifier
![image](https://github.com/Auraja/test-md/assets/116571074/b9954dde-c26f-441b-88ce-9c058c72be39)
Decision Tree Classifier

Hasil metric kedua algoritma cenderung sama, maka pengukuran di ubah menjadi menggunakan waktu eksekusi model dengan hasil berikut:
![image](https://github.com/Auraja/test-md/assets/116571074/275e8d6e-ceab-45bf-b55a-c994888881ab)
Dari data yang diberikan, terdapat dua algoritma yang dievaluasi:
1. RandomForestClassifier dengan skor 0.96 dan waktu eksekusi 1.155 detik.
2. DecisionTreeClassifier dengan skor 0.96 dan waktu eksekusi 0.008 detik.
Kedua algoritma ini memiliki skor akurasi yang sama, yaitu 0.96. Namun, jika kita mempertimbangkan waktu eksekusi, DecisionTreeClassifier jauh lebih cepat dibandingkan dengan RandomForestClassifier. Dengan demikian, berdasarkan kriteria waktu eksekusi yang lebih rendah, DecisionTreeClassifier dapat dipilih sebagai algoritma terbaik dalam hal kinerja waktu.

## Evaluation
Metric evaluasi yang digunakan adalah akurasi pada model yang digunakan dan waktu eksekusi proses pelatihan model

1. Akurasi:
Akurasi mengukur proporsi data yang diklasifikasikan dengan benar oleh model.

Rumus:

Akurasi = (Jumlah Prediksi Benar / Jumlah Data) x 100%

y_pred_tree: Prediksi kelas untuk data pengujian oleh DecisionTreeClassifier.

tree.score(X_test, y_test): Fungsi untuk menghitung akurasi model.

tree_score: Variabel yang menyimpan nilai akurasi.

3. Waktu pelatihan:
Metrik ini mengukur waktu yang dibutuhkan model untuk dilatih pada data.

Rumus:

Waktu Pelatihan = Waktu Akhir - Waktu Awal

t1: Waktu awal pelatihan model.

t2: Waktu akhir pelatihan model.

delta_tree: Variabel yang menyimpan waktu pelatihan dalam detik.

![image](https://github.com/Auraja/test-md/assets/116571074/c0822063-2c35-4e5c-8675-6a04e412e9fa)
Kedua model, RandomForestClassifier dan DecisionTreeClassifier, memiliki skor akurasi yang sama, yaitu 0.96. Namun, model DecisionTreeClassifier memiliki waktu eksekusi yang jauh lebih cepat dibandingkan RandomForestClassifier. Dengan Delta_t hanya 0.008 detik, model DecisionTreeClassifier mungkin lebih efisien dalam penggunaan sumber daya komputasi.



