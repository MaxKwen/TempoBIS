# TempoBIS — catatan temuan per fase

Semua angka di sini berasal dari output run Kaggle yang nyata. Jangan isi dengan estimasi.

## Fase 0 — Audit dataset (4 Nov 2026)

**Ringkasan:** 1.600 video, 32 gloss × 50 video, 5 signer. Tidak ada file unreadable,
duplikat ID, duplikat konten, atau nama file yang tidak ter-parse.

| Signer | Video | Gloss tersedia |
|---|---|---|
| 0 | 120 | 12 (10 video/gloss) |
| 1 | 420 | 32 (10 atau 15/gloss) |
| 2 | 420 | 32 (10 atau 15/gloss) |
| 3 | 320 | 32 (10/gloss) |
| 4 | 320 | 32 (10/gloss) |

- **Signer 0 hanya punya 12 gloss:** Air, Belajar, Cari, Hari, Ingat, Lagi, Maaf, Makan,
  Motor, Saya, Terima kasih, Tuli (`empty_label_signer_cells = 20`, semuanya di signer 0).
  Ke-12 gloss ini juga satu-satunya yang berjumlah 10 di signer 1/2 (gloss lain 15). Kemungkinan
  data direkam dalam dua tahap.
  - **Dampak:** `experiment.py:128` menghitung macro-F1 dengan `labels=range(32)` dan
    `zero_division=0`. Pada fold test = signer 0, 20 kelas yang absen otomatis F1 = 0, jadi
    macro-F1 maksimal 12/32 ≈ 0,375. Rata-rata LOSO dan uji Wilcoxon ikut bias.
  - **Rencana:** macro-F1 dihitung atas kelas yang ada di `y_test`. Fold signer 0 dilaporkan
    sebagai "12-way efektif". Rata-rata 5 fold dilaporkan bersama rata-rata 4 fold (signer 1–4).
    Confusion matrix / classification report pooled (`report.py`) tidak terdampak.
  - **Status:** belum diimplementasikan.
- **FPS:** 1.159/1.600 video persis 30 fps. Sisanya 24,87–29,98 fps dengan angka desimal yang
  tidak umum (indikasi rekaman HP / VFR). Resampling temporal berbasis indeks frame tidak
  terpengaruh. **TODO:** crosstab fps<30 × signer dan resolusi × signer untuk cek confound domain.
- **Resolusi:** 1920×1080 (1.353), 1280×720 (247).
- **Panjang klip:** 42–173 frame (mean 64,9 ± 12,5; median 63). Durasi 1,40–5,88 s
  (median 2,14 s). Minimum 42 > T = 32, jadi tidak pernah upsampling. Outlier 173 frame
  dicek di analisis error.

## Fase 1 — Pilot MediaPipe, 10 video (4 Nov 2026)

| video_id | gloss | frames | pose | lhand | rhand | any_hand |
|---|---|---|---|---|---|---|
| signer2_label0_sample8 | Air | 70 | 1.0 | 0.000 | 0.586 | 0.586 |
| signer0_label1_sample2 | Belajar | 87 | 1.0 | 0.023 | 0.609 | 0.609 |
| signer0_label2_sample7 | Cari | 57 | 1.0 | 0.018 | 0.737 | 0.737 |
| signer3_label3_sample6 | Hari | 62 | 1.0 | 1.000 | 0.694 | 1.000 |
| signer0_label4_sample4 | Ingat | 68 | 1.0 | 0.044 | 0.662 | 0.662 |
| signer0_label5_sample9 | Lagi | 58 | 1.0 | 0.069 | 0.569 | 0.569 |
| signer0_label6_sample2 | Maaf | 73 | 1.0 | 0.041 | 0.603 | 0.603 |
| signer1_label7_sample9 | Makan | 84 | 1.0 | 0.036 | 0.690 | 0.690 |
| signer4_label8_sample6 | Motor | 63 | 1.0 | 0.492 | 0.476 | 0.524 |
| signer1_label9_sample8 | Saya | 59 | 1.0 | 0.000 | 0.661 | 0.661 |

Catatan: pilot hanya mencakup 10 gloss pertama (label 0–9), dan 5 dari 10 videonya dari signer 0.

Cek visual `signer0_label1_sample2` (Belajar):
- Pose terdeteksi di semua frame. Tangan kanan (magenta) menempel di tangan yang terangkat ke dahi
  (frame 28, 57). Penetapan kiri/kanan memakai pergelangan pose terdekat, dan hasilnya benar
  (tangan kanan anatomis signer = `rhand`).
- Frame 0 dan 86 (tangan istirahat di bawah): tidak ada tangan terdeteksi.
- Titik kuning di bawah gambar = 33 landmark pose mentah (pinggul/lutut/kaki di luar frame,
  y > 1). Hanya kosmetik plot; fitur memakai 7 titik pose saja.
- Segmen aktif (tinggi pergelangan) ≈ frame 17–74, trim ≈ 14–81. Keduanya menutupi isyarat.
- Energi gerak memuncak di fase angkat (~20) dan turun (~72), hampir 0 saat tangan ditahan di dahi
  (~25–65).

Deteksi tangan di dalam vs di luar segmen aktif:

| video | active_frac | lhand in | rhand in | lhand out | rhand out |
|---|---|---|---|---|---|
| signer0_label1_sample2 | 0.67 | 0.03 | 0.91 | 0.0 | 0.00 |
| signer0_label2_sample7 | 0.79 | 0.02 | 0.93 | 0.0 | 0.00 |
| signer0_label4_sample4 | 0.71 | 0.06 | 0.94 | 0.0 | 0.00 |
| signer0_label5_sample9 | 0.62 | 0.11 | 0.92 | 0.0 | 0.00 |
| signer0_label6_sample2 | 0.66 | 0.06 | 0.92 | 0.0 | 0.00 |
| signer1_label7_sample9 | 0.67 | 0.05 | 0.98 | 0.0 | 0.11 |
| signer1_label9_sample8 | 0.64 | 0.00 | 0.95 | 0.0 | 0.14 |
| signer2_label0_sample8 | 0.66 | 0.00 | 0.89 | 0.0 | 0.00 |
| signer3_label3_sample6 | 0.65 | 1.00 | 0.62 | 1.0 | 0.82 |
| signer4_label8_sample6 | 0.56 | 0.89 | 0.86 | 0.0 | 0.00 |

- Hipotesis terbukti: tangan dominan terdeteksi 0,86–0,98 di dalam segmen aktif (9/10 video),
  dan ≈0 di luar (tangan istirahat di bawah frame). Fallback ke pergelangan pose memadai.
- `lhand` ≈ 0 pada isyarat satu tangan. Motor (dua tangan) terdeteksi 0,89/0,86.
- **Anomali:** signer3 Hari. lhand terdeteksi 1,0 di seluruh video termasuk saat istirahat,
  rhand justru lebih rendah di segmen aktif (0,62) daripada di luarnya (0,82). Dugaan: tangan
  signer 3 tetap terlihat saat istirahat (framing lebih lebar?) dan/atau kedua tangan saling
  menutupi saat isyarat. Perlu dicek visual sebelum ekstraksi penuh.
- **Hasil cek visual signer3 Hari:** framing signer 3 lebih jauh (badan kecil, kedua tangan
  terlihat di bawah saat istirahat). Itu sebabnya lhand = 1,0 dan rhand_out = 0,82 (tangan
  istirahat ikut terdeteksi). Kiri/kanan benar (magenta = tangan kanan signer, di sisi kiri
  gambar). Tangan kanan tetap terdeteksi saat diangkat (frame 20) dan saat melintang di dada
  (frame 40). rhand_in 0,62 kemungkinan dari frame yang hilang saat gerak cepat. Segmen aktif ≈ 11–50 dan trim ≈ 6–54
  menutupi isyarat. Onset aktif (11) sedikit setelah puncak gerak angkat (10), wajar untuk
  ambang tinggi pergelangan.
  - **Confound per signer:** pada signer yang tangannya keluar frame saat istirahat, fitur tangan di
    fase istirahat = fallback pergelangan, flag deteksi = 0. Pada signer 3 = bentuk tangan asli,
    flag = 1. Fitur fase istirahat dan flag deteksi bisa membawa "identitas framing" signer.
    Dicek di Fase 6 (fold signer 3). Opsi mitigasi bila perlu: samakan perlakuan tangan di luar
    segmen aktif, atau ablasi tanpa flag deteksi.
  - **Keputusan:** Fase 1 lolos. Ekstraksi penuh pakai config sekarang (lite, conf 0,5).

Interpretasi awal (sebelum dicek):
- `rhand_rate` ≈ 0,6 kemungkinan setara proporsi segmen aktif (57/87 ≈ 0,66 untuk Belajar),
  artinya tangan hilang terutama saat istirahat. Kalau benar, fallback ke pergelangan pose memadai.
  **Perlu dicek:** detection rate di dalam vs di luar segmen aktif.
- `lhand` ≈ 0 di sebagian besar video, kecuali Hari (signer 3, 1,0) dan Motor (signer 4, 0,49).
  Perlu dipastikan gloss dua tangan memang terdeteksi kedua tangannya.
- **Risiko riset `motion_aware`:** pada isyarat "tahan" seperti Belajar, densitas gerak
  menaruh sebagian besar timestep di fase angkat/turun (epentesis), bukan di pose tahan yang
  informatif. α = 0,3 menahan sebagian. Ini hipotesis yang perlu diuji di Fase 6, bukan bug.
