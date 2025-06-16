import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import time 

# Mengatur konfigurasi halaman Streamlit
st.set_page_config(page_title="Klasifikasi Buah Segar/Busuk", layout="centered")

# Fungsi untuk memuat file CSS eksternal
def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"Peringatan: File CSS '{file_name}' tidak ditemukan. Menggunakan gaya default.")
    except Exception as e:
        st.warning(f"Peringatan: Gagal memuat file CSS. Error: {e}")

# Memuat file CSS
load_css('style.css')

# Fungsi untuk memuat model TensorFlow (menggunakan st.cache_resource agar model hanya dimuat sekali)
@st.cache_resource
def load_fresh_rotten_model():
    model_path = 'fresh_rotten_classifier.h5' # Nama file model
    if not os.path.exists(model_path):
        # Menampilkan pesan error jika model tidak ditemukan dan menghentikan aplikasi
        st.error(f"Error: File model '{model_path}' tidak ditemukan. "
                 "Pastikan model 'fresh_rotten_classifier.h5' berada di direktori yang sama dengan skrip ini.")
        st.stop() 
    try:
        # Memuat model Keras dari file .h5
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        # Menampilkan pesan error jika ada masalah saat memuat model
        st.error(f"Error saat memuat model: {e}")
        st.stop()

# Memuat model klasifikasi
model_fresh_rotten = load_fresh_rotten_model()

# Ukuran target untuk gambar yang akan diprediksi (sesuai dengan input model)
IMG_TARGET_SIZE = (100, 100) 

# Fungsi untuk memuat dan mempersiapkan gambar untuk prediksi
def load_image_for_prediction(img_file):
    # Memuat gambar dan mengubah ukurannya ke IMG_TARGET_SIZE
    img = image.load_img(img_file, target_size=IMG_TARGET_SIZE)
    # Mengubah gambar menjadi array NumPy
    img_array = image.img_to_array(img)
    # Menambahkan dimensi batch (misal: dari (100, 100, 3) menjadi (1, 100, 100, 3))
    img_array = np.expand_dims(img_array, axis=0) 
    # Normalisasi nilai piksel ke rentang [0, 1]
    img_array = img_array / 255.0 
    return img_array

# Fungsi untuk melakukan prediksi apakah buah segar atau busuk
def predict_fresh_or_rotten(model, img_file):
    # Mempersiapkan gambar untuk prediksi
    img_array = load_image_for_prediction(img_file)
    # Melakukan prediksi menggunakan model
    # [0][0] diambil karena output model biner adalah skalar probabilitas
    prediction = model.predict(img_array)[0][0] 

    # Menentukan label berdasarkan nilai probabilitas (threshold 0.5)
    if prediction > 0.5:
        label = 'Busuk'
        confidence = prediction # Probabilitas menjadi busuk
    else:
        label = 'Segar' 
        confidence = 1 - prediction # Probabilitas menjadi segar (1 - probabilitas busuk)

    # Mengembalikan label dan tingkat keyakinan
    return label, confidence 

# Judul utama aplikasi
st.title("Klasifikasi Buah: Segar atau Busuk?")
# Garis pemisah untuk estetika
st.markdown("---") 

# Instruksi untuk pengguna
st.write("Unggah gambar buah Apel, Pisang dan Jeruk untuk menentukan apakah buah tersebut segar atau busuk.")

# Widget untuk mengunggah file gambar
uploaded_file = st.file_uploader("Pilih gambar buah...", type=["jpg", "jpeg", "png"])

# Logika jika file gambar telah diunggah
if uploaded_file is not None:
    # Menampilkan gambar yang diunggah oleh pengguna
    st.image(uploaded_file, caption='Gambar yang Diunggah', use_container_width=True) 
    st.write("") # Baris kosong untuk spasi

    # Menampilkan indikator loading saat model memproses
    with st.spinner('Menganalisis gambar buah...'):
        # Simulasi waktu pemrosesan model (sesuaikan jika model Anda butuh waktu lebih lama)
        time.sleep(1.5) 
        # Melakukan prediksi
        label, confidence = predict_fresh_or_rotten(model_fresh_rotten, uploaded_file)

    # Mengonversi probabilitas ke persentase
    confidence_percent = confidence * 100 

    # Menampilkan hasil prediksi berdasarkan label
    if label == 'Segar':
        st.success(f"🎉 **Hasil Prediksi:** Buah ini adalah **{label}** dengan keyakinan **{confidence_percent:.2f}%**!")
    else:
        st.error(f"⚠️ **Hasil Prediksi:** Buah ini adalah **{label}** dengan keyakinan **{confidence_percent:.2f}%**.")

# Garis pemisah di bagian bawah
st.markdown("---") 
# Footer aplikasi dengan HTML dan CSS inline
st.markdown("""
<div class="footer">
    Aplikasi Klasifikasi Buah oleh AkmalAditAlbarr | Dibuat dengan Streamlit dan TensorFlow
</div>
""", unsafe_allow_html=True)
