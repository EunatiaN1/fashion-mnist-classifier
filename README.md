# Fashion-MNIST: CNN and Random Forest

This project compares a TensorFlow convolutional neural network (CNN) with the strongest Random Forest configuration reported in the Fashion-MNIST benchmark by Xiao, Rasul, and Vollgraf (2017). The experiment uses the official 60,000-image training split and 10,000-image test split, reports per-class precision and recall, plots train/test confusion matrices, and measures model-fit time.

## Results

The executed notebook currently reports:

| Model | Train accuracy | Test accuracy | Fit time |
| --- | ---: | ---: | ---: |
| CNN, batch size 126 | 95.4% | 92.3% | 480.80 s |
| Random Forest | 100.0% | 87.5% | 38.49 s |

The CNN has the better held-out accuracy and smaller train/test gap. Both models have their lowest test recall for Shirt, most often confusing it with T-shirt/top. Training time varies by computer; rerun the notebook to reproduce measurements on your machine.

## Project files

- `Fashion_MNIST_Comparison.py` is the standalone Python program that trains both models, evaluates every split, saves confusion-matrix plots and CSV metrics, and writes your local PDF report with its code appendix.
- `Fashion_MNIST_Comparison.ipynb` contains the complete executed experiment, analysis, tables, confusion matrices, and code appendix.
- `main.py` is the separate live-camera/image demo. It loads `Model/keras_model.h5` and `Model/labels.txt`.
- `Model/` contains the small model export and class labels needed by the demo.
- `requirements.txt` lists the tested Python packages.

The live-camera demo is not the CNN trained by the comparison script. Fashion-MNIST contains 28×28 grayscale product thumbnails, not photographs of clothing in a room; benchmark accuracy should not be interpreted as real-world camera accuracy. The supplied `.tm` archive contains only T-shirt images and is not needed to run the benchmark.

## Setup

Tested with Python 3.14.6 on macOS. From the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the complete comparison from a terminal:

```bash
python Fashion_MNIST_Comparison.py
```

The script trains the CNN (10 epochs, batch size 126, learning rate 0.001) and Random Forest on the official 60,000-image training set, then evaluates on the untouched 10,000-image test set. It writes metric CSVs and plots under `outputs/` and creates `Fashion_MNIST_Report.pdf` locally. Both generated locations are ignored by Git, so the report stays on your computer. The first run downloads and caches Fashion-MNIST outside the repository.

The executed notebook is also available as a walkthrough; in VS Code, select the `.venv` Python kernel and run its cells from top to bottom.

If the automatic dataset download fails because of a local certificate issue, download the four files listed in the Fashion-MNIST repository's [Get the Data section](https://github.com/zalandoresearch/fashion-mnist#get-the-data) into `~/.keras/datasets/fashion-mnist/`, then rerun the notebook.

## Live-camera demo

The demo opens camera 0 when run without arguments:

```bash
python main.py
```

Press `q` or Escape to close the preview. To classify an image file instead, run `python main.py "/path/to/image.jpg"`. This demo is provided separately from the benchmark comparison.

## References

Xiao, H., Rasul, K., & Vollgraf, R. (2017). *Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms*. https://doi.org/10.48550/arXiv.1708.07747

TensorFlow. (2024). *Basic classification: Classify images of clothing*. https://www.tensorflow.org/tutorials/keras/classification