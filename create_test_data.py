import csv
from pathlib import Path
import numpy as np

# Create synthetic dataset for testing
data_file = Path('C:/Users/KOMALI/Downloads/Exams/sign-language-recognition') / 'data' / 'raw' / 'gesture_data.csv'
data_file.parent.mkdir(parents=True, exist_ok=True)

gestures = ['HELLO', 'YES', 'NO', 'THANK_YOU']
samples_per_gesture = 10

with open(data_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['gesture', 'landmarks'])
    
    for gesture in gestures:
        for i in range(samples_per_gesture):
            # Create random 21x3 landmarks (realistic range: 0-1)
            landmarks = np.random.uniform(0, 1, 63)
            landmarks_str = ' '.join(str(x) for x in landmarks)
            writer.writerow([gesture, landmarks_str])

total = len(gestures) * samples_per_gesture
print(f'Created synthetic dataset with {len(gestures)} gestures and {total} samples')
print(f'Location: {data_file}')
