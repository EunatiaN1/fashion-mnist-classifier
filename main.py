import argparse
from pathlib import Path

from keras import Sequential
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D, Input
import cv2
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)


parser = argparse.ArgumentParser(description="Classify a fashion product image.")
parser.add_argument("image", nargs="?", type=Path, help="Path to an image to classify")
parser.add_argument(
	"--camera",
	nargs="?",
	const=0,
	type=int,
	metavar="INDEX",
	help="Classify live camera video (default camera index: 0)",
)
args = parser.parse_args()
if args.image is not None and args.camera is not None:
	parser.error("provide an image path or use --camera, but not both")
if args.image is None and args.camera is None:
	args.camera = 0

model_dir = Path(__file__).resolve().parent / "Model"
class_names = [line.strip().split(maxsplit=1)[1] for line in (model_dir / "labels.txt").read_text().splitlines()]

# Rebuild the legacy nested model in current Keras, then load its trained weights.
backbone = MobileNetV2(input_shape=(224, 224, 3), alpha=0.35, include_top=False, weights=None)
feature_extractor = Sequential(
	[backbone, GlobalAveragePooling2D(name="global_average_pooling2d_GlobalAveragePooling2D2")],
	name="sequential_5",
)
classifier = Sequential(
	[
		Input(shape=(backbone.output_shape[-1],), name="dense_Dense3_input"),
		Dense(100, activation="relu", name="dense_Dense3"),
		Dense(len(class_names), activation="softmax", use_bias=False, name="dense_Dense4"),
	],
	name="sequential_7",
)
model = Sequential(
	[Input(shape=(224, 224, 3), name="sequential_5_input"), feature_extractor, classifier],
	name="sequential_8",
)
model.load_weights(model_dir / "keras_model.h5")

def classify(image):
	image = ImageOps.fit(image.convert("RGB"), (224, 224), Image.Resampling.LANCZOS)
	data = np.asarray(image, dtype=np.float32)[np.newaxis, ...]
	data = (data / 127.5) - 1
	prediction = model.predict(data, verbose=0)[0]
	index = int(np.argmax(prediction))
	return class_names[index], float(prediction[index])


def classify_from_camera(camera_index):
	camera = cv2.VideoCapture(camera_index)
	if not camera.isOpened():
		camera.release()
		raise SystemExit(
			f"Could not open camera {camera_index}. Check macOS camera permission "
			"for Terminal or VS Code, or try another index with --camera INDEX."
		)

	try:
		while True:
			ok, frame = camera.read()
			if not ok:
				raise SystemExit("The camera opened but did not return a frame.")

			rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			class_name, confidence = classify(Image.fromarray(rgb_frame))
			cv2.putText(
				frame,
				f"{class_name} ({confidence:.1%})",
				(16, 36),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.9,
				(40, 220, 80),
				2,
				cv2.LINE_AA,
			)
			cv2.imshow("Fashion classifier", frame)
			if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
				break
	finally:
		camera.release()
		cv2.destroyAllWindows()


if args.camera is not None:
	classify_from_camera(args.camera)
else:
	with Image.open(args.image) as image:
		class_name, confidence = classify(image)
	print("Class:", class_name)
	print("Confidence Score:", confidence)
