"""Train and compare a Fashion-MNIST CNN and Random Forest.

Run this script from the project environment to retrain both models, print
metrics, save plots and CSV tables, and generate a local PDF report with the
complete source code appended. The PDF is ignored by Git and stays local.
"""

import argparse
import random
import time
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as PdfImage,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def parse_args():
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Compare a TensorFlow CNN and a Random Forest on Fashion-MNIST."
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=126)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=project_dir / "outputs")
    parser.add_argument(
        "--report",
        type=Path,
        default=project_dir / "Fashion_MNIST_Report.pdf",
        help="Local PDF output path (ignored by Git by default).",
    )
    return parser.parse_args()


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def train_cnn(x_train, y_train, x_test, epochs, batch_size, learning_rate):
    from tensorflow.keras import layers

    model = tf.keras.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(2),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(2),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(10),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    started = time.perf_counter()
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=2,
    )
    fit_seconds = time.perf_counter() - started
    train_pred = np.argmax(model.predict(x_train, batch_size=512, verbose=0), axis=1)
    test_pred = np.argmax(model.predict(x_test, batch_size=512, verbose=0), axis=1)
    return model, history, train_pred, test_pred, fit_seconds


def train_random_forest(x_train, y_train, x_test, seed):
    x_train_flat = x_train.reshape(len(x_train), -1).astype(np.float32)
    x_test_flat = x_test.reshape(len(x_test), -1).astype(np.float32)
    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=100,
            criterion="entropy",
            max_depth=50,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        ),
    )
    started = time.perf_counter()
    model.fit(x_train_flat, y_train)
    fit_seconds = time.perf_counter() - started
    train_pred = model.predict(x_train_flat)
    test_pred = model.predict(x_test_flat)
    return model, train_pred, test_pred, fit_seconds


def evaluate_models(predictions, y_train, y_test):
    rows = []
    class_rows = []
    matrices = {}
    labels = np.arange(len(CLASS_NAMES))

    for model_name, split_predictions in predictions.items():
        for split_name, y_pred in split_predictions.items():
            y_true = y_train if split_name == "Train" else y_test
            matrix = confusion_matrix(y_true, y_pred, labels=labels)
            matrices[(model_name, split_name)] = matrix
            macro = precision_recall_fscore_support(
                y_true, y_pred, labels=labels, average="macro", zero_division=0
            )
            weighted = precision_recall_fscore_support(
                y_true, y_pred, labels=labels, average="weighted", zero_division=0
            )
            rows.append(
                {
                    "Model": model_name,
                    "Split": split_name,
                    "Accuracy": accuracy_score(y_true, y_pred),
                    "Macro precision": macro[0],
                    "Macro recall": macro[1],
                    "Macro F1": macro[2],
                    "Weighted precision": weighted[0],
                    "Weighted recall": weighted[1],
                    "Weighted F1": weighted[2],
                }
            )
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, labels=labels, zero_division=0
            )
            for index, class_name in enumerate(CLASS_NAMES):
                class_rows.append(
                    {
                        "Model": model_name,
                        "Split": split_name,
                        "Class": class_name,
                        "Precision": precision[index],
                        "Recall": recall[index],
                        "F1": f1[index],
                        "Support": support[index],
                    }
                )

    return pd.DataFrame(rows), pd.DataFrame(class_rows), matrices


def save_plots(history, matrices, output_dir):
    epochs = np.arange(1, len(history.history["accuracy"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history.history["accuracy"], marker="o", color="#2463a6")
    axes[0].set(title="CNN training accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[0].set_ylim(0, 1)
    axes[1].plot(epochs, history.history["loss"], marker="o", color="#c45a27")
    axes[1].set(title="CNN training loss", xlabel="Epoch", ylabel="Loss")
    fig.tight_layout()
    learning_curve = output_dir / "learning_curves.png"
    fig.savefig(learning_curve, dpi=160, bbox_inches="tight")
    plt.close(fig)

    labels = np.arange(len(CLASS_NAMES))
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    for ax, key in zip(axes.flat, matrices):
        model_name, split_name = key
        matrix = matrices[key]
        image = ax.imshow(matrix, cmap="Blues")
        threshold = matrix.max() / 2
        for row, column in np.ndindex(matrix.shape):
            value = matrix[row, column]
            ax.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                fontsize=6,
                color="white" if value > threshold else "#17212b",
            )
        ax.set_title(f"{model_name} - {split_name}")
        ax.set_xlabel("Predicted category")
        ax.set_ylabel("True category")
        ax.set_xticks(labels, CLASS_NAMES, rotation=45, ha="right")
        ax.set_yticks(labels, CLASS_NAMES)
        ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.7)
        ax.tick_params(which="minor", bottom=False, left=False)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    confusion_plot = output_dir / "confusion_matrices.png"
    fig.savefig(confusion_plot, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return learning_curve, confusion_plot


def report_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            alignment=TA_CENTER,
            fontSize=22,
            leading=27,
            textColor=colors.HexColor("#1c5e91"),
            spaceAfter=18,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading1"],
            fontSize=15,
            leading=18,
            textColor=colors.HexColor("#1c5e91"),
            spaceBefore=4,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SubHeading",
            parent=styles["Heading2"],
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#263746"),
            spaceBefore=8,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportBody",
            parent=styles["BodyText"],
            fontSize=9,
            leading=13,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CodeAppendix",
            fontName="Courier",
            fontSize=5.7,
            leading=7,
            leftIndent=0,
            rightIndent=0,
        )
    )
    return styles


def paragraph(text, style):
    return Paragraph(escape(text), style)


def make_table(data, widths, font_size=8):
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1c5e91")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("LEADING", (0, 0), (-1, -1), font_size + 2),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5df")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f6f9")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def create_report(
    report_path,
    summary,
    class_metrics,
    matrices,
    cnn_seconds,
    forest_seconds,
    args,
    output_dir,
):
    styles = report_styles()
    body = styles["ReportBody"]
    story = [
        paragraph("Fashion-MNIST Classification: CNN vs Random Forest", styles["ReportTitle"]),
        paragraph("Abstract", styles["SectionHeading"]),
        paragraph(
            "This project compares a TensorFlow convolutional neural network (CNN) with a "
            "Random Forest on the standard Fashion-MNIST benchmark. The script trains both "
            "models on the official training split, measures fit time, and evaluates train "
            "and test predictions with confusion matrices and precision/recall tables.",
            body,
        ),
        paragraph("1. The dataset", styles["SectionHeading"]),
        paragraph(
            "Fashion-MNIST contains 70,000 grayscale images of clothing products. Each image "
            "is 28 by 28 pixels and belongs to one of ten categories: T-shirt/top, Trouser, "
            "Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, or Ankle boot. The canonical "
            "split has 60,000 training images and 10,000 test images, with 6,000 and 1,000 "
            "examples per class respectively.",
            body,
        ),
        paragraph(
            "These are small product silhouettes rather than everyday camera photographs. "
            "That makes the dataset useful for a controlled comparison, but results should "
            "not be read as a guarantee that a model will recognize a garment held up to a webcam.",
            body,
        ),
        paragraph("2. Experimental setup", styles["SectionHeading"]),
        paragraph(
            f"The official split was kept intact and the test set was reserved for final "
            f"evaluation. A fixed seed of {args.seed} controls random initialization where "
            f"the libraries allow it. CNN pixel values are scaled from 0-255 to 0-1. Forest "
            f"images are flattened to 784 features; a StandardScaler is fitted on training "
            f"data only and then applied to both splits.",
            body,
        ),
        PageBreak(),
        paragraph("3. CNN implementation", styles["SectionHeading"]),
        paragraph(
            "The CNN learns patterns from the image grid rather than treating each pixel as "
            "unrelated. It has two 3x3 convolution layers (32 and 64 filters), each followed "
            "by 2x2 max pooling, then a flattened feature vector, a 128-unit ReLU layer, "
            "dropout at 0.3, and a ten-logit output layer.",
            body,
        ),
        paragraph(
            f"Training used Adam with learning rate {args.learning_rate:g}, sparse categorical "
            f"cross-entropy from logits, {args.epochs} epochs, batch size {args.batch_size}, "
            "and shuffled training examples. The 2024 TensorFlow clothing tutorial provides "
            "the dataset loader and preprocessing/training approach; its example is fully "
            "connected, so convolution and pooling are added here to meet the CNN requirement.",
            body,
        ),
        paragraph("4. Random Forest implementation", styles["SectionHeading"]),
        paragraph(
            "The forest follows the highest-scoring RandomForestClassifier configuration in "
            "the authors' published benchmark grid: entropy split criterion, maximum depth "
            "50, and 100 trees. The remaining settings use scikit-learn defaults except for "
            "a fixed random seed and parallel tree fitting across available CPU cores. The "
            "images are flattened; unlike the CNN, the forest does not directly preserve "
            "the neighborhood relationship between pixels.",
            body,
        ),
        PageBreak(),
        paragraph("5. Results", styles["SectionHeading"]),
        paragraph(
            "The table reports accuracy and macro precision, recall, and F1 on both splits. "
            "Macro averages give each category equal weight, which is useful when comparing "
            "the ten balanced classes. The separate class table later in the report gives "
            "per-category precision and recall.",
            body,
        ),
    ]

    summary_data = [[
        "Model", "Split", "Accuracy", "Macro P", "Macro R", "Macro F1"
    ]]
    for row in summary.itertuples(index=False):
        summary_data.append(
            [
                row.Model,
                row.Split,
                f"{row.Accuracy:.3f}",
                f"{row._3:.3f}",
                f"{row._4:.3f}",
                f"{row._5:.3f}",
            ]
        )
    story.append(make_table(summary_data, [1.35 * inch, 0.65 * inch, 0.8 * inch, 0.8 * inch, 0.8 * inch, 0.8 * inch]))
    story.extend(
        [
            Spacer(1, 10),
            paragraph(
                f"Fit time on this machine was {cnn_seconds:.2f} seconds for the CNN and "
                f"{forest_seconds:.2f} seconds for the forest. These timings are hardware- "
                "and load-dependent; they measure fitting, not prediction latency.",
                body,
            ),
            PdfImage(str(output_dir / "learning_curves.png"), width=6.8 * inch, height=2.5 * inch),
            PageBreak(),
            paragraph("6. Precision and recall by category", styles["SectionHeading"]),
            paragraph(
                "Precision shows how often a category prediction is right. Recall shows how "
                "many real examples of that category the model found. The table includes both "
                "measures on training and test data, along with F1 and sample support.",
                body,
            ),
        ]
    )

    class_data = [["Model", "Split", "Category", "Precision", "Recall", "F1", "N"]]
    for row in class_metrics.itertuples(index=False):
        class_data.append(
            [row.Model, row.Split, row.Class, f"{row.Precision:.3f}", f"{row.Recall:.3f}", f"{row.F1:.3f}", str(row.Support)]
        )
    story.extend(
        [
            make_table(
                class_data,
                [1.0 * inch, 0.55 * inch, 1.15 * inch, 0.72 * inch, 0.72 * inch, 0.62 * inch, 0.45 * inch],
                font_size=7,
            ),
            PageBreak(),
            paragraph("7. Confusion matrices", styles["SectionHeading"]),
            paragraph(
                "Rows show the true class and columns show the predicted class. Numbers on "
                "the diagonal are correct predictions; off-diagonal numbers show the errors.",
                body,
            ),
            PdfImage(str(output_dir / "confusion_matrices.png"), width=7.0 * inch, height=5.6 * inch),
            PageBreak(),
        ]
    )

    error_notes = []
    for model_name in predictions_from_matrices(matrices):
        test_rows = class_metrics[
            (class_metrics["Model"] == model_name) & (class_metrics["Split"] == "Test")
        ]
        worst = test_rows.loc[test_rows["Recall"].idxmin()]
        matrix = matrices[(model_name, "Test")].copy()
        np.fill_diagonal(matrix, 0)
        true_index, predicted_index = np.unravel_index(np.argmax(matrix), matrix.shape)
        error_notes.append(
            f"{model_name}: lowest test recall was {worst['Class']} at {worst['Recall']:.1%}. "
            f"Its largest single confusion was {CLASS_NAMES[true_index]} predicted as "
            f"{CLASS_NAMES[predicted_index]} ({matrix[true_index, predicted_index]} images)."
        )

    cnn_test = summary.query("Model == 'CNN' and Split == 'Test'").iloc[0]
    forest_test = summary.query("Model == 'Random Forest' and Split == 'Test'").iloc[0]
    cnn_train = summary.query("Model == 'CNN' and Split == 'Train'").iloc[0]
    forest_train = summary.query("Model == 'Random Forest' and Split == 'Train'").iloc[0]
    story.extend(
        [
            paragraph("8. What the errors tell us", styles["SectionHeading"]),
            paragraph(" ".join(error_notes), body),
            paragraph(
                "Shirt and T-shirt/top are easy to mix up in these thumbnails. At 28 by 28 "
                "pixels, the images have no color, fabric texture, or real-world context; "
                "several upper-body garments share a similar outline. The CNN can learn "
                "local shapes from neighboring pixels, while the forest sees a flat feature "
                "vector. That helps explain the difference, though neither model removes the "
                "ambiguity in the images themselves.",
                body,
            ),
            paragraph("9. Recommendation", styles["SectionHeading"]),
            paragraph(
                f"For this benchmark I recommend the CNN when held-out accuracy matters most. "
                f"It reached {cnn_test['Accuracy']:.1%} test accuracy, compared with "
                f"{forest_test['Accuracy']:.1%} for the forest. The forest fit much faster, "
                f"but its {forest_train['Accuracy']:.1%} training score fell on test data to "
                f"{forest_test['Accuracy']:.1%}, a clear sign of overfitting. The CNN's train "
                f"and test scores were {cnn_train['Accuracy']:.1%} and {cnn_test['Accuracy']:.1%}. "
                "For real camera use, I would collect representative photographs and evaluate "
                "them separately before deployment.",
                body,
            ),
            PageBreak(),
            paragraph("10. Limits and references", styles["SectionHeading"]),
            paragraph(
                "These results describe grayscale product thumbnails, not clothes in a live "
                "camera scene. The project's camera demo is separate from the benchmark models. "
                "The supplied Teachable Machine archive contains T-shirt images only and is "
                "not used in this ten-class experiment. Timings vary with hardware and system load.",
                body,
            ),
            paragraph(
                "The Random Forest is a benchmark baseline rather than a tuned modern production "
                "system. The comparison uses one fixed seed and one official train/test split. "
                "The test set was not used to choose parameters.",
                body,
            ),
            paragraph(
                "Xiao, H., Rasul, K., &amp; Vollgraf, R. (2017). <i>Fashion-MNIST: A novel image "
                "dataset for benchmarking machine learning algorithms</i>. "
                "<link href='https://doi.org/10.48550/arXiv.1708.07747'>doi:10.48550/arXiv.1708.07747</link>.",
                body,
            ),
            paragraph(
                "TensorFlow. (2024). <i>Basic classification: Classify images of clothing</i>. "
                "Retrieved December 10, 2025, from "
                "<link href='https://www.tensorflow.org/tutorials/keras/classification'>tensorflow.org/tutorials/keras/classification</link>.",
                body,
            ),
            paragraph(
                "Random Forest parameter grid and benchmark results: "
                "<link href='https://github.com/zalandoresearch/fashion-mnist'>authors' repository</link>.",
                body,
            ),
            PageBreak(),
            paragraph("Appendix A. Complete Python code", styles["SectionHeading"]),
            paragraph(
                "The source used to generate this report follows. The PDF is written locally; "
                "it is excluded from Git by the project's ignore rules.",
                body,
            ),
        ]
    )

    source = Path(__file__).read_text(encoding="utf-8")
    story.append(Preformatted(source, styles["CodeAppendix"], maxLineLength=115))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(report_path),
        pagesize=letter,
        leftMargin=0.62 * inch,
        rightMargin=0.62 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title="Fashion-MNIST CNN and Random Forest Comparison",
        author="Fashion-MNIST Classification Project",
    )
    document.build(story)


def predictions_from_matrices(matrices):
    return list(dict.fromkeys(model_name for model_name, _ in matrices))


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise SystemExit("Epochs and batch size must be positive; learning rate must be greater than zero.")

    set_seeds(args.seed)
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report = args.report.expanduser().resolve()

    print("Loading the canonical Fashion-MNIST train/test split...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    assert x_train.shape == (60_000, 28, 28)
    assert x_test.shape == (10_000, 28, 28)
    x_train_cnn = x_train.astype(np.float32)[..., np.newaxis] / 255.0
    x_test_cnn = x_test.astype(np.float32)[..., np.newaxis] / 255.0

    print("Training CNN...")
    cnn_model, history, cnn_train_pred, cnn_test_pred, cnn_seconds = train_cnn(
        x_train_cnn,
        y_train,
        x_test_cnn,
        args.epochs,
        args.batch_size,
        args.learning_rate,
    )

    print("Training Random Forest...")
    forest_model, forest_train_pred, forest_test_pred, forest_seconds = train_random_forest(
        x_train,
        y_train,
        x_test,
        args.seed,
    )

    predictions = {
        "CNN": {"Train": cnn_train_pred, "Test": cnn_test_pred},
        "Random Forest": {"Train": forest_train_pred, "Test": forest_test_pred},
    }
    summary, class_metrics, matrices = evaluate_models(predictions, y_train, y_test)
    summary.to_csv(args.output_dir / "summary_metrics.csv", index=False)
    class_metrics.to_csv(args.output_dir / "class_metrics.csv", index=False)
    learning_curve, confusion_plot = save_plots(history, matrices, args.output_dir)
    create_report(
        args.report,
        summary,
        class_metrics,
        matrices,
        cnn_seconds,
        forest_seconds,
        args,
        args.output_dir,
    )

    print("\nOverall metrics:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nPer-class precision and recall:")
    print(
        class_metrics[["Model", "Split", "Class", "Precision", "Recall", "Support"]]
        .to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )
    print("\nTraining times:")
    print(f"CNN: {cnn_seconds:.2f} seconds")
    print(f"Random Forest: {forest_seconds:.2f} seconds")
    print(f"\nSaved plots: {learning_curve} and {confusion_plot}")
    print(f"Saved private local PDF report: {args.report}")


if __name__ == "__main__":
    main()