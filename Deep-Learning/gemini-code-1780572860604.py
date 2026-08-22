import tensorflow as tf
from tensorflow.keras import layers, models

# =====================================================================
# 1. TABULAR BRANCH (Financial Ratios, Bureau Metrics)
# =====================================================================
# Input shape reflects the number of numeric/engineered tabular features
tabular_input = layers.Input(shape=(15,), name="tabular_input")

x_tab = layers.Dense(64, activation="relu")(tabular_input)
x_tab = layers.BatchNormalization()(x_tab)
x_tab = layers.Dropout(0.3)(x_tab)
x_tab = layers.Dense(32, activation="relu")(x_tab)


# =====================================================================
# 2. TEXT BRANCH (Loan Descriptions, Purpose Text)
# =====================================================================
# Input accepts integer tokens (e.g., from TextVectorization layer)
text_input = layers.Input(shape=(100,), dtype="int32", name="text_input")

# Map integer tokens to dense 64-dimensional vectors
x_text = layers.Embedding(input_dim=10000, output_dim=64, name="text_embeddings")(text_input)

# Compress the sequence dimension (100, 64) down to a flat vector (64,)
x_text = layers.GlobalAveragePooling1D()(x_text) 

x_text = layers.Dense(32, activation="relu")(x_text)


# =====================================================================
# 3. CONCATENATION & DEEP HEAD
# =====================================================================
# Merge the 32-dim tabular output and 32-dim text output into a 64-dim vector
merged_features = layers.Concatenate()([x_tab, x_text])

# Final joint reasoning layers
x_joint = layers.Dense(32, activation="relu")(merged_features)
x_joint = layers.Dropout(0.2)(x_joint)

# Binary classification output for credit default (0 = Good, 1 = Default)
output = layers.Dense(1, activation="sigmoid", name="default_prediction")(x_joint)


# =====================================================================
# 4. MODEL COMPILATION
# =====================================================================
# Explicitly map multi-inputs and single output
model = models.Model(inputs=[tabular_input, text_input], outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.AUC(name="auc", curve="ROC"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

# Inspect model topology and parameter tracking
model.summary()