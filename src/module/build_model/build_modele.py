import mlflow
import mlflow.tensorflow
from mlflow.models import infer_signature
import tensorflow as tf


# Démarrer une nouvelle exécution
with mlflow.start_run():
    # Suivi automatique de TensorFlow
    mlflow.tensorflow.autolog()

    # Exemple d'entraînement de modèle
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(8,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    # Entraîner le modèle
    model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

    # Suivi des métriques personnalisées
    mlflow.log_param("batch_size", 32)
    mlflow.log_metric("final_loss", model.evaluate(X_test, y_test))

    # Enregistrer le modèle dans MLflow
    mlflow.tensorflow.log_model(model, "model")


tuner = kt.Hyperband(
    build_model,
    objective='val_root_mean_squared_error',
    max_epochs=epochs,
    factor=3,
    directory='../vehicle_prediction',  # Dossier pour enregistrer les résultats
    project_name='vehicle_prediction'
)

# Callback pour enregistrer les résultats dans MLflow
mlflow_callback = tf.keras.callbacks.LambdaCallback(
    on_epoch_end=lambda epoch, logs: mlflow.log_metrics(logs, step=epoch)
)

# Définir un callback pour arrêter la recherche si la performance ne s'améliore pas
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Lancer la recherche des hyperparamètres
tuner.search(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=[stop_early, mlflow_callback])

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Enregistrer les meilleurs hyperparamètres dans MLflow
mlflow.log_params({
    'dense_1_units': best_hps.get('dense_1_units'),
    'dense_2_units': best_hps.get('dense_2_units'),
    'dropout_rate': best_hps.get('dropout_rate'),
    'learning_rate': best_hps.get('learning_rate')
})

print(f"""
La recherche d'hyperparamètres est terminée. Le nombre optimal d'unités dans la première
couche densément connectée est {best_hps.get('units')} et le taux d'apprentissage optimal pour l'optimiseur
est {best_hps.get('learning_rate')}.
""")