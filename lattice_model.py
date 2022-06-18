# %%
import tensorflow as tf
import logging
import numpy as np
import pandas as pd
import sys
import tensorflow_lattice as tfl
from tensorflow import feature_column as fc
import matplotlib.pyplot as plt
logging.disable(sys.maxsize)

# %%
X_train,y_train = pd.read_csv('./data/processed/X_train.csv',index_col=0),pd.read_csv('./data/processed/y_train.csv',index_col=0)
X_train,y_train = pd.read_csv('./data/processed/X_train.csv',index_col=0),pd.read_csv('./data/processed/y_train.csv',index_col=0)


X_train=X_train.fillna(-1000)
# %%

lattice_sizes = [3,3,3,3]

model_inputs = []
lattice_inputs = []

# ############### value ###############
value_input = tf.keras.layers.Input(shape=[1], name='value')
model_inputs.append(value_input)
value_calibrator = tfl.layers.PWLCalibration(
    # Every PWLCalibration layer must have keypoints of piecewise linear
    # function specified. Easiest way to specify them is to uniformly cover
    # entire input range by using numpy.linspace().
    input_keypoints=np.linspace(
        X_train['value'].min(), X_train['value'].max(), num=10),
    # You need to ensure that input keypoints have same dtype as layer input.
    # You can do it by setting dtype here or by providing keypoints in such
    # format which will be converted to desired tf.dtype by default.
    dtype=tf.float32,
    # Output range must correspond to expected lattice input range.
    output_min=0.0,
    output_max=lattice_sizes[0] - 1.0,
    monotonicity='increasing',
    name='value_calib',
    impute_missing=True,
    missing_input_value=-1000,
    kernel_regularizer=('wrinkle', 0.0, 1e-4),
)(
    value_input)
lattice_inputs.append(value_calibrator)

# ############### ict_index_lag1 ###############
ict_index_lag1_input = tf.keras.layers.Input(shape=[1], name='ict_index_lag1')
model_inputs.append(ict_index_lag1_input)
ict_index_lag1_calibrator = tfl.layers.PWLCalibration(
    # Every PWLCalibration layer must have keypoints of piecewise linear
    # function specified. Easiest way to specify them is to uniformly cover
    # entire input range by using numpy.linspace().
    input_keypoints=np.linspace(
        X_train['ict_index_lag1'].min(), X_train['ict_index_lag1'].max(), num=10),
    # You need to ensure that input keypoints have same dtype as layer input.
    # You can do it by setting dtype here or by providing keypoints in such
    # format which will be converted to desired tf.dtype by default.
    dtype=tf.float32,
    # Output range must correspond to expected lattice input range.
    output_min=0.0,
    output_max=lattice_sizes[1] - 1.0,
    monotonicity='increasing',
    name='ict_index_lag1_calib',
    impute_missing=True,
    missing_input_value=-1000,
    kernel_regularizer=('wrinkle', 0.0, 1e-4),

)(
    ict_index_lag1_input)
lattice_inputs.append(ict_index_lag1_calibrator)

# ############### ict_index_lag1 ###############
minutes_lag1_input = tf.keras.layers.Input(shape=[1], name='minutes_lag1')
model_inputs.append(minutes_lag1_input)
minutes_lag1_calibrator = tfl.layers.PWLCalibration(
    # Every PWLCalibration layer must have keypoints of piecewise linear
    # function specified. Easiest way to specify them is to uniformly cover
    # entire input range by using numpy.linspace().
    input_keypoints=np.linspace(
        X_train['minutes_lag1'].min(), X_train['minutes_lag1'].max(), num=10),
    # You need to ensure that input keypoints have same dtype as layer input.
    # You can do it by setting dtype here or by providing keypoints in such
    # format which will be converted to desired tf.dtype by default.
    dtype=tf.float32,
    # Output range must correspond to expected lattice input range.
    output_min=0.0,
    output_max=lattice_sizes[1] - 1.0,
    monotonicity='increasing',
    name='minutes_lag1_calib',
    impute_missing=True,
    missing_input_value=-1000,
    kernel_regularizer=('wrinkle', 0.0, 1e-4),

)(
    minutes_lag1_input)
lattice_inputs.append(minutes_lag1_calibrator)

# ############### ict_index_lag1 ###############
total_points_lag1_input = tf.keras.layers.Input(shape=[1], name='total_points_lag1')
model_inputs.append(total_points_lag1_input)
total_points_lag1_calibrator = tfl.layers.PWLCalibration(
    # Every PWLCalibration layer must have keypoints of piecewise linear
    # function specified. Easiest way to specify them is to uniformly cover
    # entire input range by using numpy.linspace().
    input_keypoints=np.linspace(
        X_train['total_points_lag1'].min(), X_train['total_points_lag1'].max(), num=10),
    # You need to ensure that input keypoints have same dtype as layer input.
    # You can do it by setting dtype here or by providing keypoints in such
    # format which will be converted to desired tf.dtype by default.
    dtype=tf.float32,
    # Output range must correspond to expected lattice input range.
    output_min=0.0,
    output_max=lattice_sizes[1] - 1.0,
    monotonicity='increasing',
    name='total_points_lag1_calib',
    impute_missing=True,
    missing_input_value=-1000,
    kernel_regularizer=('wrinkle', 0.0, 1e-2),

)(
    total_points_lag1_input)
lattice_inputs.append(total_points_lag1_calibrator)

###
lattice = tfl.layers.Lattice(
    lattice_sizes=lattice_sizes,
    monotonicities=[
        'increasing', 'increasing','increasing','increasing'
    ],
    output_min=0.0,
    output_max=1.0,
    name='lattice',
)(lattice_inputs)

model_output = tf.keras.layers.Dense(1)(lattice)

model = tf.keras.models.Model(
    inputs=model_inputs,
    outputs=model_output)
tf.keras.utils.plot_model(model, rankdir='LR')



# %%
feature_names = ['value', 'ict_index_lag1','minutes_lag1','total_points_lag1']
features = np.split(
    X_train[feature_names].values.astype(np.float32),
    indices_or_sections=len(feature_names),
    axis=1)
target = y_train.values.astype(np.float32)

val_features = np.split(
    X_train[feature_names].values.astype(np.float32),
    indices_or_sections=len(feature_names),
    axis=1)
val_target = y_train.values.astype(np.float32)

model.compile(
    loss=tf.keras.losses.mean_squared_error,
    optimizer=tf.keras.optimizers.Adam(1e-1))
model.fit(
    features,
    target,
    batch_size=32,
    epochs=500,
    validation_split=0.2,
    shuffle=False,
    verbose=1)

# %%
f1_mesh = xlist = np.linspace(X_train.value[lambda x:x>0].min(), max(features[0])*1.2, 100)
f2_mesh = xlist = np.linspace(X_train.ict_index_lag1[lambda x:x>0].min(), max(features[1])*1.2, 100)
f3_mesh = xlist = np.linspace(X_train.minutes_lag1[lambda x:x>0].min(), max(features[2])*1.2, 100)
f4_mesh = xlist = np.linspace(X_train.total_points_lag1[lambda x:x>0].min(), max(features[3])*1.2, 100)

f1_def = np.full((10000, 1), X_train.value.quantile(0.75))
f2_def = np.full((10000, 1), X_train.ict_index_lag1.quantile(0.75))
f3_def = np.full((10000, 1), X_train.minutes_lag1.quantile(0.75))
f4_def = np.full((10000, 1), X_train.total_points_lag1.quantile(0.75))

X, Y = np.meshgrid(f3_mesh, f4_mesh )
prediction = model.predict([f1_def,f2_def,X.reshape(-1,1),Y.reshape(-1,1)])

Z= prediction.reshape(X.shape)


# %%
fig,ax=plt.subplots(1,1)
cp = ax.contourf(X, Y, Z)
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('Filled Contours Plot')
ax.set_xlabel('Minutes')
ax.set_ylabel('Total Points L1')
plt.show()
#X_out = X_train.copy()
#X_out['pred'] = preds

# %%
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import train_test_split, GridSearchCV

numeric_features = feature_names
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
    ]
)

clf = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", RidgeCV(scoring='neg_mean_squared_error'))]
)

clf.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error
lreg_pred = clf.predict(X_train)
print("model score: %.3f" % mean_squared_error(y_train,lreg_pred))
# %%
