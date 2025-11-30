from django.shortcuts import render
from .forms import UploadForm
from .processing import spatial_processing, frequency_processing, extract_text
import cv2
import os
from django.conf import settings

def index(request):
    context = {}

    if request.method == "POST":
        form = UploadForm(request.POST, request.FILES)

        if form.is_valid():
            image_file = form.cleaned_data['image']

            # Save original
            img_path = os.path.join(settings.MEDIA_ROOT, "input.jpg")
            with open(img_path, "wb") as f:
                for chunk in image_file.chunks():
                    f.write(chunk)

            img = cv2.imread(img_path)

            # Spatial
            spatial = spatial_processing(img)
            spatial_path = os.path.join(settings.MEDIA_ROOT, "spatial.jpg")
            cv2.imwrite(spatial_path, spatial)

            # Frequency
            freq = frequency_processing(img)
            freq_path = os.path.join(settings.MEDIA_ROOT, "freq.jpg")
            cv2.imwrite(freq_path, freq)

            # Final combine
            combined = cv2.addWeighted(spatial, 0.6, freq, 0.4, 0)
            final_path = os.path.join(settings.MEDIA_ROOT, "final.jpg")
            cv2.imwrite(final_path, combined)

            # OCR
            extracted_text = extract_text(combined)

            context = {
                'spatial_img': "spatial.jpg",
                'freq_img': "freq.jpg",
                'final_img': "final.jpg",
                'text': extracted_text
            }
    else:
        form = UploadForm()

    context['form'] = form
    return render(request, "index.html", context)
