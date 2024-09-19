import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
import mlflow.keras
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dropout, GlobalAveragePooling2D, Dense
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from timeit import default_timer as timer
from monitoring.alert_system import AlertSystem
from app.utils.logger import setup_logger
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logger = setup_logger("train_model", f"logs/train_model_{timestamp}.log")


class TimingCallback(Callback):
    def __init__(self, logs={}):
        self.logs = []

    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer() - self.starttime)


def get_latest_experiment_id(experiment_name):
    experiments = mlflow.search_experiments(filter_string=f"name='{experiment_name}'")
    if experiments:
        return experiments[0].experiment_id
    else:
        mlflow.create_experiment(experiment_name)
        return mlflow.get_experiment_by_name(experiment_name).experiment_id


def train_model(start_mlflow_run=True, data_version=None, experiment_id=None):
    if experiment_id is None:
        experiment_id = get_latest_experiment_id("Model Training Experiment")

    if not mlflow.active_run():
        with mlflow.start_run(experiment_id=experiment_id, run_name="Model Training", nested=True):
            return _train_model_internal(data_version, experiment_id)
    else:
        return _train_model_internal(data_version, experiment_id)


def _train_model_internal(data_version, experiment_id):
    try:
        dataset_path = os.path.join(BASE_DIR, "data")
        train_path = os.path.join(dataset_path, "train")
        valid_path = os.path.join(dataset_path, "valid")
        test_path = os.path.join(dataset_path, "test")

        for path in [dataset_path, train_path, valid_path, test_path]:
            if not os.path.exists(path):
                logger.error(f"Le dossier {path} n'existe pas.")
                raise FileNotFoundError(f"Le dossier {path} n'existe pas.")

        logger.info(f"Chemin d'entraînement : {train_path}")
        logger.info(f"Chemin actuel : {os.getcwd()}")

        batch_size = 16
        mlflow.log_param("batch_size", batch_size)

        reduce_learning_rate = ReduceLROnPlateau(
            monitor="val_loss",
            patience=2,
            min_delta=0.01,
            factor=0.1,
            cooldown=4,
            verbose=1,
        )
        early_stopping = EarlyStopping(patience=5, min_delta=0.01, verbose=1, mode="min", monitor="val_loss")
        time_callback = TimingCallback()

        train_datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode="nearest",
        )

        train_generator = train_datagen.flow_from_directory(
            train_path,
            target_size=(224, 224),
            batch_size=batch_size
        )
        valid_generator = ImageDataGenerator().flow_from_directory(
            valid_path,
            target_size=(224, 224),
            batch_size=batch_size
        )
        test_generator = ImageDataGenerator().flow_from_directory(
            test_path,
            target_size=(224, 224),
            batch_size=batch_size
        )

        num_classes = train_generator.num_classes
        mlflow.log_param("num_classes", num_classes)

        base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

        for layer in base_model.layers[:-20]:
            layer.trainable = False
        for layer in base_model.layers[-20:]:
            layer.trainable = True

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1280, activation="relu")(x)
        x = Dropout(rate=0.2)(x)
        x = Dense(640, activation="relu")(x)
        x = Dropout(rate=0.2)(x)
        predictions = Dense(num_classes, activation="softmax")(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        optimizer = Adam(learning_rate=0.001)
        mlflow.log_param("initial_learning_rate", 0.001)
        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["acc", "mean_absolute_error"],
        )

        training_history = model.fit(
            train_generator,
            epochs=1,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            validation_data=valid_generator,
            validation_steps=valid_generator.samples // valid_generator.batch_size,
            callbacks=[reduce_learning_rate, early_stopping, time_callback],
            verbose=1,
        )

        test_loss, test_accuracy, test_mae = model.evaluate(test_generator)

        logger.info(f"Test accuracy: {test_accuracy}")
        logger.info(f"Final validation accuracy: {training_history.history['val_acc'][-1]}")

        os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
        model_save_path = os.path.join(BASE_DIR, "models", f"saved_model_{timestamp}")
        try:
            tf.saved_model.save(model, model_save_path)
            mlflow.log_artifact(model_save_path, artifact_path="model")
            logger.info(f"Model saved successfully at {model_save_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise

        mlflow.log_metric("test_accuracy", float(test_accuracy))
        mlflow.log_metric("test_loss", float(test_loss))
        mlflow.log_metric("test_mae", float(test_mae))
        mlflow.log_metric("final_val_accuracy", float(training_history.history["val_acc"][-1]))

        for epoch, (acc, val_acc, loss, val_loss) in enumerate(
            zip(
                training_history.history["acc"],
                training_history.history["val_acc"],
                training_history.history["loss"],
                training_history.history["val_loss"],
            )
        ):
            mlflow.log_metrics(
                {
                    f"accuracy_epoch_{epoch+1}": float(acc),
                    f"val_accuracy_epoch_{epoch+1}": float(val_acc),
                    f"loss_epoch_{epoch+1}": float(loss),
                    f"val_loss_epoch_{epoch+1}": float(val_loss),
                },
                step=epoch,
            )

        for epoch, time in enumerate(time_callback.logs):
            mlflow.log_metric(f"epoch_{epoch+1}_time", float(time), step=epoch)

        drift_detected = False
        if experiment_id:
            client = mlflow.tracking.MlflowClient()
            runs = client.search_runs(
                experiment_ids=[experiment_id],
                filter_string="metrics.test_accuracy > 0 and tags.run_type = 'training'",
                order_by=["attribute.start_time DESC"],
                max_results=2,
            )

            if len(runs) > 1:
                previous_run = runs[1]
                previous_accuracy = previous_run.data.metrics["test_accuracy"]

                if previous_accuracy > test_accuracy:
                    performance_drop = previous_accuracy - test_accuracy
                    if performance_drop > 0.05:
                        drift_detected = True
                        alert_message = f"Dégradation des performances détectée. \
                        Ancienne accuracy: {previous_accuracy}, Nouvelle accuracy: {test_accuracy}"
                        alert_system = AlertSystem()
                        alert_system.send_alert("Alerte de Dégradation des Performances", alert_message)
                        logger.warning(alert_message)

                mlflow.log_metric("previous_test_accuracy", previous_accuracy)
                mlflow.log_metric("accuracy_change", test_accuracy - previous_accuracy)

        return model, drift_detected

    except Exception as e:
        logger.error(f"Une erreur s'est produite lors de l'entraînement du modèle : {str(e)}")
        raise


if __name__ == "__main__":
    train_model()
