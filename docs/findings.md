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

## Fase 2 — Ekstraksi penuh (4 Nov 2026)

Config: `pose_model: lite`, `min_detection_confidence: 0.5`, `min_tracking_confidence: 0.5`.
Sempat dihentikan di ≈850 video, lalu dilanjutkan (resume).

- **1600/1600 video ter-extract.** `landmark_stats.csv` dibangun dengan me-load setiap `.npz`,
  jadi tidak ada file korup (kalau ada, `np.load` akan error).
- Statistik `frames` identik dengan `decoded_frames` di audit (mean 64,888125; min 42; median 63;
  max 173). Setiap file berisi seluruh frame video, tidak ada yang terpotong.

| | pose_rate | lhand_rate | rhand_rate | any_hand_rate |
|---|---|---|---|---|
| mean | 1.0 | 0.164 | 0.648 | 0.665 |
| std | 0.0 | 0.295 | 0.122 | 0.142 |
| min | 1.0 | 0.000 | 0.212 | 0.212 |
| 25% | 1.0 | 0.000 | 0.560 | 0.567 |
| 50% | 1.0 | 0.024 | 0.652 | 0.657 |
| 75% | 1.0 | 0.082 | 0.727 | 0.737 |
| max | 1.0 | 1.000 | 0.949 | 1.000 |

- Pose 100% di semua video.
- `rhand_rate` mean 0,65 konsisten dengan pilot (tangan hilang saat istirahat, bukan gagal deteksi).
- `lhand_rate` median 0,02: mayoritas isyarat satu tangan; ekor atas (max 1,0) kemungkinan
  signer 3 (framing lebar) + gloss dua tangan.
- **TODO:** cek video dengan `rhand_rate` rendah (min 0,21) dan rata-rata per signer × gloss,
  serta `rhand_rate` di dalam segmen aktif.
- **Backup:** `/kaggle/working/outputs` (landmarks/, audit/, pilot/, landmark_stats.csv) diunggah
  sebagai Kaggle Dataset privat `maximusquinnhertada/tempobis-landmarks` (CC BY-NC 4.0) via
  `kaggle datasets create --dir-mode zip`.

## Fase 3 — Quick run (protokol `si_signer4`, 1 seed)

Clean macro-F1 (core): uniform 0,625 · trim_uniform 0,771 · motion_aware 0,676.
Drop core: 0,005 / 0,001 / 0,003. n = 1, tanpa uji statistik. Pipeline belajar jauh di atas 1/32.

## Fase 4–5 — Eksperimen penuh (GRU, LOSO 5 fold × 3 seed)

**Peringatan:** metrik masih memakai `labels=range(32)`, jadi fold signer 0 dibatasi maksimal 12/32.
Semua rata-rata di bawah ikut bias ke bawah, dan std antar-fold (±0,29) membengkak.

Clean macro-F1, mean ± std atas fold:
uniform 0,605 ± 0,289 · trim_uniform 0,639 ± 0,290 · motion_aware 0,608 ± 0,299 ·
uniform_speedaug 0,630 ± 0,304.

Drop (clean − rata-rata terperturbasi):
- global: 0,001 / −0,000 / 0,000 / −0,000
- core: 0,000 / 0,000 / 0,000 / 0,002
- rentang core uniform di semua faktor: 0,598–0,611

Paired vs uniform, n = 15 pasangan (fold, seed):
- Δclean: trim +0,034 (p = 0,0084) · motion_aware +0,003 (p = 0,80) · speedaug +0,025 (p = 0,107)
- Δdrop: semuanya |Δ| ≤ 0,002, p ≥ 0,15

Run per fold yang terlihat di log (macro-F1 clean, uniform/trim/motion/speedaug):
- signer 2 seed 2: 0,870 / 0,889 / 0,906 / 0,924
- signer 3 seed 0–2: 0,271–0,342 / 0,310–0,444 / 0,275–0,308 / 0,286–0,415
- signer 4 seed 0–2: 0,645–0,726 / 0,733–0,772 / 0,739–0,768 / 0,743–0,778

**Interpretasi sementara:**
1. Pada rentang 0,5–1,5×, perturbasi kecepatan hampir tidak menurunkan performa metode mana pun,
   termasuk baseline. Hipotesis "motion_aware lebih tahan" tidak bisa diuji karena baseline
   tidak terganggu. Perlu dipastikan dulu bahwa perturbasi core benar-benar mengubah input.
2. Trim memberi kenaikan clean yang konsisten. Tapi n = 15 berisi 3 seed per fold (tidak
   independen). Dengan uji tingkat fold (n = 5), p minimum dua sisi = 0,0625.
3. Fold signer 3 jauh lebih rendah (≈0,27–0,44). Konsisten dengan confound framing
   (tangan istirahat terlihat) dari Fase 1.

### Diagnosis setelah Fase 5

Akurasi clean (faktor 1,0), rata-rata 3 seed:

| test_signer | motion_aware | trim_uniform | uniform | uniform_speedaug |
|---|---|---|---|---|
| 0 | 0.764 | 0.769 | 0.803 | 0.750 |
| 1 | 0.833 | 0.886 | 0.876 | 0.885 |
| 2 | 0.909 | 0.914 | 0.893 | 0.911 |
| 3 | 0.357 | 0.429 | 0.378 | 0.410 |
| 4 | 0.764 | 0.766 | 0.691 | 0.771 |

Perturbasi vs input uniform (rata-rata |Δ| fitur setelah resample T = 32, per signer):

| signer | fallback | core_frac | d_core 0,5× | d_core 1,5× | d_global 0,5× |
|---|---|---|---|---|---|
| 0 | 0.0 | 0.649 | 0.098 | 0.071 | 0.0 |
| 1 | 0.0 | 0.694 | 0.093 | 0.071 | 0.0 |
| 2 | 0.0 | 0.710 | 0.089 | 0.068 | 0.0 |
| 3 | 0.0 | 0.676 | 0.099 | 0.074 | 0.0 |
| 4 | 0.0 | 0.636 | 0.096 | 0.069 | 0.0 |

- **Bug metrik terkonfirmasi:** akurasi fold signer 0 = 0,75–0,80, padahal macro-F1-nya dibatasi ≤ 0,375.
  Fold yang benar-benar sulit hanya signer 3 (0,36–0,43).
- `active_segment` tidak pernah fallback. Segmen isyarat ≈ 64–71% klip.
- Perturbasi global terhadap uniform = 0 persis. Ini sesuai desain (pemeriksaan kewajaran):
  resample linier dari resample linier di posisi yang sama.
- Perturbasi core memang mengubah input (≈0,07–0,10), tapi model hampir tidak terpengaruh.
  Skala 0,07–0,10 belum punya pembanding (mis. jarak antar-sampel kelas sama / antar-kelas).
- Kesimpulan sementara: dengan uniform resampling + GRU, perubahan kecepatan segmen isyarat
  0,5–1,5× bukan masalah nyata di dataset ini. Belum ada "kerentanan" untuk diperbaiki
  motion_aware.

### Perubahan setelah diagnosis (belum dijalankan di Kaggle)

- **Metrik:** macro-F1 sekarang dirata-rata atas kelas yang ada di test signer
  (`experiment.macro_f1_present`). `results.csv` mencatat `n_test_classes` dan `n_classes`.
  `summary.md` menandai fold yang cakupan kelasnya berkurang dan menampilkan clean macro-F1
  dengan dan tanpa fold tersebut. **Angka Fase 4–5 di atas memakai metrik lama.**
- **Stress test lebih berat** (`configs/stress.yaml`, hasil ke `outputs/results_stress/`):
  faktor 0,25–4×, ditambah `warp` (setengah awal isyarat s×, setengah akhir 1/s×) dan `hold`
  (jeda di puncak isyarat, tinggi pergelangan maksimum; hanya s < 1).
- Test lokal: 32 lulus, 1 skip (GRU, butuh PyTorch).
