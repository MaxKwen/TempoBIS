# Deploying the demo (Hugging Face Spaces)

Do this only after the Kaggle experiment has finished.

1. From `tempobis_models.zip` (Kaggle output), pick one model trained with the proposed method,
   e.g. `results/models/test_signer4_seed0_motion_aware.pt`. Any fold works; using the official
   signer-4 fold keeps it easy to explain.
2. Create a new Space at huggingface.co → *New Space* → SDK **Gradio**, CPU basic (free).
3. Upload to the Space, keeping this repo's folder layout:
   * `src/tempobis/` (whole folder)
   * `demo/app.py`
   * the model → `demo/model_motion_aware.pt` (the method is read from the file name)
   * `results/config_used.json` → `demo/config_used.json`
   * `demo/requirements.txt` → `requirements.txt` (Space root)
   * a `README.md` at the Space root whose header points to the app:
     ```yaml
     ---
     title: TempoBIS
     sdk: gradio
     app_file: demo/app.py
     ---
     ```
4. The first launch downloads the MediaPipe model files (needs internet, allowed on Spaces).
5. Test with a few WL-BISINDO videos from the **held-out** signer of that fold, and with yourself
   on webcam. Note in the Space description that it was trained on 5 Banten signers.

Dataset license is CC BY-NC 4.0: keep the Space non-commercial, credit WL-BISINDO, and do not
host the dataset videos in the Space.
