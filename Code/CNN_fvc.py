# Linear Regression for Slope per patient -> label

# Each slice per patient predicts slope

# Compute FVC at week 52 

# Decline based on FVC at 0 or FVC at first available value



class SliceLevelDataset(Dataset):

  def __init__(self, df, root_path, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_path = root_path
        self.transform = transform
        self.samples = []

        for idx, row in df.iterrows():
          patient = row['Patient']
          event = row['event']
          time = row['time']
          slice_files = row['slice_files']

          for slice_file in slice_files:
            self.samples.append({
                'slice_path':os.path.join(root_path,patient,slice_file),
                'patient':patient,
                'event':event,
                'time':time
            })

        print(f"Dataset: {len(df)} patients → {len(self.samples)} slices")

  def __len__(self):
        return len(self.samples)


  def preprocess_slice(self, img):
      """Usa preprocessing corretto"""

      ds = pydicom.dcmread(img)
      img_raw = ds.pixel_array.astype(np.float32)


      _, _, _, _, mask, _, _, _ = make_lungmask(img_raw.copy())
      img_masked = img_raw * mask

      # Lung window
      img_windowed = np.clip(img_masked, -1000, 400)
      img_normalized = (img_windowed + 1000) / 1400

      # Resize
      img_resized = cv2.resize(img_normalized, (224, 224))

      # RGB
      img_rgb = np.stack([img_resized] * 3, axis=0)

      # ImageNet normalization
      mean = np.array([0.485, 0.456, 0.406])[:, None, None]
      std = np.array([0.229, 0.224, 0.225])[:, None, None]
      img_final = (img_rgb - mean) / std

      return img_final, mask

  def __getitem__(self, idx):
      sample = self.samples[idx]


      try:
          # Preprocess the image and get the mask
          img, mask = self.preprocess_slice(sample['slice_path'])

          # Check if the image is "empty" or has no meaningful content
          # Soglia per la presenza del polmone (ad esempio, almeno 100 pixel del polmone)
          if np.sum(mask) < 30:  # Soglia di pixel validi del polmone
              raise ValueError("Immagine senza polmone significativo.")

      except Exception as e:
          return None  # Return None to skip this sample

      # Return a valid sample
      return {
          'image': torch.FloatTensor(img),
          'patient': sample['patient'],
          'event': torch.tensor(sample['event'], dtype=torch.float32),
          'time': torch.tensor(sample['time'], dtype=torch.float32)
      }


class SliceLevelCNN(nn.Module):

    #Add here clinical_Dim to insert handcrafted features
    def __init__(self, backbone_name='efficientnet_b0',pretrained = True):
      super().__init__()

      

    def forward(self, x):

      



